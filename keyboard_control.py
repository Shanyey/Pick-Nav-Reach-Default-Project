import heapq
from copy import deepcopy
from enum import Enum, auto

import numpy as np
import pybullet as p


class GraspState(Enum):
    IDLE = auto()

    # FINAL ATTEMPT STATES
    WAITING = auto()
    CANDIDATES = auto()
    SOLVE_IK = auto()
    APPROACHING = auto()
    ADJUSTING_GRIPPER = auto()
    FINAL_APPROACH = auto()
    CLOSING = auto()
    HOLDING = auto()
    RETURN_TO_START = auto()
    NAVIGATING = auto()
    FAILED = auto()


class KeyBoardController:
    def __init__(self, env):
        ### GENERAL
        self.env = env
        self.action = env.agent_qpos  # np.zeros(self.env.action_size)
        # self.action_scale = np.array([0.03] * env.action_size)
        # self.action_scale[:3] = 0.02 # root movement
        # self.action_scale[-2:] = 0.005  # gripper
        self.action_scale = np.array([0.03] * env.action_size)
        self.action_scale[:3] = 0.04  # root movement
        self.action_scale[-2:] = 0.01  # gripper
        self.action_upper_bound = env.joint_upper
        self.action_lower_bound = env.joint_lower
        self.target_action = env.agent_qpos.copy()
        self.state = GraspState.WAITING
        self.marker_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0.1, 1.0, 0.1, 0.9],
        )

        self.gripper_link_index = next(
            j
            for j in range(p.getNumJoints(self.env.robot_id))
            if p.getJointInfo(self.env.robot_id, j)[12].decode() == "gripper_link"
        )  # index 20
        self.right_gripper_link_index = next(
            j
            for j in range(p.getNumJoints(self.env.robot_id))
            if p.getJointInfo(self.env.robot_id, j)[12].decode()
            == "r_gripper_finger_link"
        )
        self.left_gripper_link_index = next(
            j
            for j in range(p.getNumJoints(self.env.robot_id))
            if p.getJointInfo(self.env.robot_id, j)[12].decode()
            == "l_gripper_finger_link"
        )
        wrist_roll_link_index = next(
            j
            for j in range(p.getNumJoints(self.env.robot_id))
            if p.getJointInfo(self.env.robot_id, j)[12].decode() == "wrist_roll_link"
        )

        ### WAITING
        self.waiting_steps = 0
        self.waiting_steps_required = 5
        self.curr_obj_pos = None
        self.object_movement_threshold = 1e-4

        ### GENERATING CANDIDATES
        self.grasp_candidates = []
        self.final_approach_action = None
        self.final_approach_still_steps = 0
        self.start_closing = False

        ### SOLVING IK
        self.max_top_to_midpoint_distance = 0.05
        self.target_position = []
        self.target_midpoint = ()
        self.target_orn = p.getQuaternionFromEuler([0, np.pi / 2, 0])

        ### APPROACHING
        self.movement_speed_threshold = 0.3
        self.target_position_tolerance = 0.02
        self.still_steps = 0
        self.still_steps_required = 3

        ### NAVIGATION
        self.waypoints = []
        self.entrance_approach_waypoints = []
        self.navigation_waypoints = []
        self.navigation_waypoint_index = 0
        self.navigation_started = False
        root_position, root_orientation = p.getBasePositionAndOrientation(
            self.env.robot_id,
            self.env.pb_physics_client,
        )
        self.robot_root_position = np.asarray(root_position, dtype=float)
        self.robot_root_yaw = float(p.getEulerFromQuaternion(root_orientation)[2])
        self.navigation_waypoint_tolerance = 0.12
        self.navigation_heading_tolerance = 0.08
        self.navigation_grid_resolution = 0.25
        self.navigation_safety_margin = 0.05
        self.final_approach = False
        self.test_flag = False

        self.closing_finger_qpos = None
        self.closing_stable_steps = 0
        self.closing_stable_steps_required = 5
        self.closing_stability_threshold = 1e-4

        self.holding = False
        self.returning_to_start = False

        self.safe_qpos = [
            6.6184002e-06,
            8.4158637e-07,
            1.9702206e-08,
            -9.3034389e-05,
            -5.5017704e-28,
            -2.1032671e-13,
            -1.2896187e-05,
            -1.2210194e00,
            -3.3208999e-05,
            6.2491822e-01,
            7.0792349e-04,
            2.1556127e00,
            2.5561580e-05,
            -1.7096805e-04,
            2.4484846e-04,
        ]
        self.ideal_grasp_qpos = [
            -1.73500027e-07,
            6.73842067e-07,
            -2.71663474e-08,
            -1.13534967e-04,
            2.69898787e-28,
            -3.55052012e-12,
            -2.05587730e-06,
            -4.59455371e-01,
            -2.49372190e-03,
            1.44456765e-02,
            2.51946901e-03,
            2.00252628e00,
            -4.67113859e-05,
            0.05,
            0.05,
        ]

    #################################### WAITING ####################################
    def handle_waiting(self, obs):
        next_obj_pos = np.asarray(obs["object_pos"], dtype=float)
        if self.curr_obj_pos is None:
            self.curr_obj_pos = next_obj_pos.copy()
            return

        movement = np.linalg.norm(next_obj_pos - self.curr_obj_pos)
        if movement <= self.object_movement_threshold:
            self.waiting_steps += 1
        else:
            self.waiting_steps = 0

        self.curr_obj_pos = next_obj_pos.copy()

        if self.waiting_steps > self.waiting_steps_required:
            self.state = GraspState.CANDIDATES
            self.waiting_steps = 0
            self.spawn_qpos = np.asarray(obs["qpos"], dtype=float)

    #################################### GRASP CANDIDATES ####################################
    def handle_candidate_generation(self, obs):
        pc, normals = obs["object_pc"], obs["object_normals"]
        candidates = self.get_all_candidates(pc, normals)
        if not candidates:
            candidates = self.get_all_candidates(pc, normals, normal_threshold=0.8)

        if candidates:
            self.visualize_grasp_candidates(candidates, length=1)
            self.grasp_candidates = candidates
            self.state = GraspState.SOLVE_IK

            self.final_approach_action = None
            self.final_approach_still_steps = 0

            self.start_closing = False
            self.closing_finger_qpos = None
            self.closing_stable_steps = 0

    def get_all_candidates(self, point_cloud, normals, normal_threshold=0.95):
        point_cloud = np.asarray(point_cloud, dtype=float)
        normals = np.asarray(normals, dtype=float)

        # Midpoint of object
        object_min = point_cloud.min(axis=0)
        object_max = point_cloud.max(axis=0)
        object_center = (object_min + object_max) / 2.0
        object_mid_z = object_center[2]

        if len(point_cloud) != len(normals):
            raise ValueError("point_cloud and normals must have the same length")

        candidates = []
        min_width, max_width = 0.004, 0.10
        eps = 1e-12

        num_points = len(point_cloud)

        for i in range(num_points):
            for j in range(i + 1, num_points):
                p1 = point_cloud[i]
                p2 = point_cloud[j]
                n1 = normals[i]
                n2 = normals[j]

                if p1[2] < object_mid_z or p2[2] < object_mid_z:
                    continue

                direction = p2 - p1
                width = np.linalg.norm(direction)
                if width < min_width or width > max_width or width < eps:
                    continue

                # Normalize the contact-point direction and normals.
                d = direction / width

                n1_length = np.linalg.norm(n1)
                n2_length = np.linalg.norm(n2)
                if n1_length < eps or n2_length < eps:
                    continue

                n1 = n1 / n1_length
                n2 = n2 / n2_length

                inward_n1 = -n1
                inward_n2 = -n2

                toward_p2 = np.dot(inward_n1, d)
                toward_p1 = np.dot(inward_n2, -d)

                if toward_p2 < normal_threshold or toward_p1 < normal_threshold:
                    continue

                quality = min(toward_p2, toward_p1)

                grasp_center = (p1 + p2) / 2.0
                center_distance = np.linalg.norm(grasp_center - object_center)
                top_to_midpoint_distance = object_max[2] - grasp_center[2]

                candidates.append(
                    {
                        "indices": (i, j),
                        "points": (p1, p2),
                        "normals": (n1, n2),
                        "width": width,
                        "quality": quality,
                        "center_distance": center_distance,
                        "top_to_midpoint_distance": top_to_midpoint_distance,
                    }
                )

        candidates.sort(key=lambda candidate: candidate["quality"], reverse=True)

        top_10 = candidates[:10]
        if not top_10:
            return []
        top_10.sort(key=lambda c: (c["center_distance"], -c["quality"]))
        return top_10

    def visualize_grasp_candidates(self, candidates, length=None):
        candidate_points = []
        if length == None:
            top_candidates = candidates
        else:
            top_candidates = candidates[:length]

        for candidate in top_candidates:
            p1, p2 = candidate["points"]
            candidate_points.extend([p1, p2])

        if not candidate_points:
            return

        p.addUserDebugPoints(
            pointPositions=np.asarray(candidate_points, dtype=float).tolist(),
            pointColorsRGB=[[1, 0, 0]] * len(candidate_points),
            pointSize=6,
            lifeTime=0,
        )

    #################################### SOLVE IK METHODS ####################################
    def handle_solve_ik(self):
        solution = self.solve_ik()  # is a list/
        if solution:
            self.target_action = np.asarray(solution, dtype=np.float32)
            self.target_action[-2:] = [1, 1]
            self.target_action = np.clip(
                self.target_action,
                self.action_lower_bound,
                self.action_upper_bound,
            )
            self.state = GraspState.APPROACHING

    def get_orientation_v2(self, candidate):
        top_to_midpoint_distance = candidate["top_to_midpoint_distance"]
        verticle_orn = p.getQuaternionFromEuler([0, np.pi / 2, 0])
        if top_to_midpoint_distance > self.max_top_to_midpoint_distance:
            print(
                "Grasp midpoint is too far below the object top: "
                f"{top_to_midpoint_distance:.3f} m"
            )
            return verticle_orn  # TODO get an angle perpendicular to the grasp

        return verticle_orn

    def get_perpendicular_directions(self, candidate):
        p1 = np.asarray(candidate["points"][0], dtype=float)
        p2 = np.asarray(candidate["points"][1], dtype=float)

        midpoint = (p1 + p2) / 2.0

        # Unit vector along the line between the two contact points.
        closing_axis = p2 - p1
        length = np.linalg.norm(closing_axis)
        if length < 1e-12:
            raise ValueError("Candidate points are too close together")

        closing_axis /= length

        # Choose a coordinate axis that is not parallel to closing_axis.
        reference_axis = np.eye(3)[np.argmin(np.abs(closing_axis))]

        # Create two perpendicular unit vectors spanning the circle's plane.
        u = np.cross(closing_axis, reference_axis)
        u /= np.linalg.norm(u)

        v = np.cross(closing_axis, u)
        v /= np.linalg.norm(v)

        # Sample the circle at 1-degree intervals: 0 through 359 degrees.
        angles = np.deg2rad(np.arange(360))
        directions = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v

        return midpoint, directions

    def _gripper_link_target_position(self, contact_position, orientation):
        rotation = np.asarray(
            p.getMatrixFromQuaternion(orientation),
            dtype=float,
        ).reshape(3, 3)
        # The 0.16645 m wrist-to-gripper offset is already in the URDF fixed joint.
        link_to_contact_offset = np.array([0.0, 0.015425, 0.0])
        return (
            np.asarray(contact_position, dtype=float)
            - rotation @ link_to_contact_offset
        )

    def solve_ik(self):
        candidate = self.grasp_candidates[0]
        if candidate is not None:
            midpoint = (candidate["points"][0] + candidate["points"][1]) / 2
            self.target_midpoint = midpoint
            self.target_position = (midpoint + np.array([0.0, 0.0, 0.1])).tolist()
            target_orn = self.get_orientation_v2(candidate)
            self.target_orn = target_orn

        lower_limits = self.action_lower_bound.copy()
        upper_limits = self.action_upper_bound.copy()
        new_target_position = self._gripper_link_target_position(
            self.target_position,
            target_orn,
        ).tolist()
        solution = p.calculateInverseKinematics(
            self.env.robot_id,
            self.gripper_link_index,
            targetPosition=new_target_position,
            maxNumIterations=1000,
            residualThreshold=1e-4,
            targetOrientation=target_orn,
            lowerLimits=lower_limits.tolist(),
            upperLimits=upper_limits.tolist(),
            jointRanges=self.env.joint_ranges.tolist(),
            restPoses=self.ideal_grasp_qpos,
        )
        self.pre_approach_link_position = new_target_position

        solution = solution[: self.env.action_size]
        return solution

    #################################### APPROACHING ####################################
    def handle_approaching(self, obs):
        print("APPROACHING PRE-GRASP LOCATION")
        qvel = np.asarray(obs["qvel"])

        if np.all(np.abs(qvel) < self.movement_speed_threshold):
            print(f"still step {self.still_steps}")
            self.still_steps += 1
        else:
            print(f"not still: {qvel}")
            self.still_steps = 0

        if self.still_steps >= self.still_steps_required:
            print("Robot stopped; checking position")

            qpos = np.asarray(obs["qpos"])
            error = self.target_action - qpos
            print(f"error: {error}")
            wrap_mask = np.isclose(self.action_upper_bound, 314.0)
            error[wrap_mask] = (error[wrap_mask] + np.pi) % (2 * np.pi) - np.pi

            if np.all(np.abs(error) <= self.target_position_tolerance):
                print("Position correct, starting closing")
                self.state = GraspState.ADJUSTING_GRIPPER
            else:
                print(f"Position incorrect, restarting pre-grasping: {error}")
                self.state = GraspState.IDLE

            self.still_steps = 0

    #################################### GRIPPER ADJUSTMENT ####################################
    def handle_gripper_adjustment(self, obs):
        correction = self.get_wrist_roll_correction(self.grasp_candidates[0])

        # The correction is an angle in radians; ignore tiny residuals.
        correction_tolerance = np.deg2rad(1.0)
        if abs(correction) <= correction_tolerance:
            print("Wrist alignment is close enough, starting final approach.")
            self.state = GraspState.FINAL_APPROACH
            return

        wrist_roll_index = 12

        # Set a new target relative to the measured wrist angle, rather than
        # accumulating correction onto the previous target every frame.
        self.target_action[wrist_roll_index] = (
            obs["qpos"][wrist_roll_index] + correction
        )
        self.target_action = np.clip(
            self.target_action,
            self.action_lower_bound,
            self.action_upper_bound,
        )

        print(f"Wrist-roll correction: {np.rad2deg(correction):.2f} degrees")

    def get_wrist_roll_correction(self, candidate):
        robot_id = self.env.robot_id

        wrist_roll_link_index = next(
            j
            for j in range(p.getNumJoints(robot_id))
            if p.getJointInfo(robot_id, j)[12].decode() == "wrist_roll_link"
        )

        right_state = p.getLinkState(
            robot_id,
            self.right_gripper_link_index,
            computeForwardKinematics=True,
        )
        left_state = p.getLinkState(
            robot_id,
            self.left_gripper_link_index,
            computeForwardKinematics=True,
        )
        wrist_state = p.getLinkState(
            robot_id,
            wrist_roll_link_index,
            computeForwardKinematics=True,
        )

        right_position = np.asarray(right_state[4], dtype=float)
        left_position = np.asarray(left_state[4], dtype=float)

        # Current gripper closing direction.
        current_direction = left_position - right_position
        current_direction /= np.linalg.norm(current_direction)

        # Candidate closing direction.
        candidate_direction = np.asarray(
            candidate["points"][0], dtype=float
        ) - np.asarray(candidate["points"][1], dtype=float)
        candidate_direction /= np.linalg.norm(candidate_direction)

        # The two candidate points have no inherent left/right ordering.
        # Choose the sign closest to the current gripper direction.
        if np.dot(current_direction, candidate_direction) < 0:
            candidate_direction = -candidate_direction

        # wrist_roll_joint rotates around the wrist_roll_link local X axis.
        wrist_orientation = np.asarray(wrist_state[5], dtype=float)
        rotation_matrix = np.asarray(
            p.getMatrixFromQuaternion(wrist_orientation)
        ).reshape(3, 3)

        wrist_axis_world = rotation_matrix @ np.array([1.0, 0.0, 0.0])
        wrist_axis_world /= np.linalg.norm(wrist_axis_world)

        # Project both directions onto the plane perpendicular to the wrist axis.
        current_projected = (
            current_direction
            - np.dot(current_direction, wrist_axis_world) * wrist_axis_world
        )
        candidate_projected = (
            candidate_direction
            - np.dot(candidate_direction, wrist_axis_world) * wrist_axis_world
        )

        current_projected /= np.linalg.norm(current_projected)
        candidate_projected /= np.linalg.norm(candidate_projected)

        signed_angle = np.arctan2(
            np.dot(
                wrist_axis_world,
                np.cross(current_projected, candidate_projected),
            ),
            np.dot(current_projected, candidate_projected),
        )

        return signed_angle

    #################################### CLOSING ####################################
    def handle_final_approach(self, obs):
        if self.final_approach is True:
            qvel = np.asarray(obs["qvel"])
            if np.all(np.abs(qvel) < self.movement_speed_threshold):
                print(f"still step {self.still_steps}")
                self.still_steps += 1
            else:
                print(f"not still: {qvel}")
                self.still_steps = 0

            if self.still_steps >= self.still_steps_required:
                self.still_steps = 0
                self.state = GraspState.CLOSING
            return

        current_qpos = np.asarray(obs["qpos"], dtype=float)
        lower_limits = self.action_lower_bound.copy()
        upper_limits = self.action_upper_bound.copy()
        midpoint = self.target_midpoint
        midpoint[-1] += 0.02  # raise the target position slightly above the midpoint
        solution = p.calculateInverseKinematics(
            self.env.robot_id,
            self.gripper_link_index,
            targetPosition=self.target_midpoint,
            maxNumIterations=1000,
            residualThreshold=1e-4,
            lowerLimits=lower_limits.tolist(),
            upperLimits=upper_limits.tolist(),
            jointRanges=self.env.joint_ranges.tolist(),
            restPoses=current_qpos.tolist(),
            physicsClientId=self.env.pb_physics_client,
        )

        self.target_action = np.asarray(
            solution[: self.env.action_size], dtype=np.float32
        )
        self.final_approach = True

    def handle_closing(self, obs):
        if not self.start_closing:
            self.start_closing = True
            self.target_action[-2:] = self.action_lower_bound[-2:]
            self.closing_finger_qpos = np.asarray(
                obs["qpos"][-2:],
                dtype=float,
            ).copy()
            self.closing_stable_steps = 0
            print("Closing grippers.")
            return

        finger_positions = np.asarray(obs["qpos"][-2:], dtype=float)

        movement = np.max(np.abs(finger_positions - self.closing_finger_qpos))
        self.closing_finger_qpos = finger_positions.copy()
        if movement <= self.closing_stability_threshold:
            self.closing_stable_steps += 1
        else:
            self.closing_stable_steps = 0
        self.closing_finger_qpos = finger_positions.copy()

        if self.closing_stable_steps < self.closing_stable_steps_required:
            return

        contact_links = {
            contact[3]
            for contact in p.getContactPoints(
                bodyA=self.env.robot_id,
                bodyB=self.env.object_id,
                physicsClientId=self.env.pb_physics_client,
            )
        }
        fingers_contacting = {
            self.left_gripper_link_index,
            self.right_gripper_link_index,
        }.issubset(contact_links)
        if fingers_contacting:
            print("Finger positions stabilized with object contact; grasp detected.")
            self.state = GraspState.HOLDING

    #################################### HOLDING ####################################
    def handle_holding(self, obs):
        if self.holding:
            qvel = np.asarray(obs["qvel"])
            if np.all(np.abs(qvel) < self.movement_speed_threshold):
                print(f"still step {self.still_steps}")
                self.still_steps += 1
            else:
                print(f"not still: {qvel}")
                self.still_steps = 0

            if self.still_steps >= self.still_steps_required:
                self.still_steps = 0
                self.state = GraspState.RETURN_TO_START
            return

        self.target_action[-2:] = self.action_lower_bound[-2:]  # Keep gripper closed
        current_qpos = np.asarray(obs["qpos"], dtype=float)
        lower_limits = self.action_lower_bound.copy()
        upper_limits = self.action_upper_bound.copy()
        current_gripper_position = np.asarray(
            p.getLinkState(
                self.env.robot_id,
                self.gripper_link_index,
                computeForwardKinematics=True,
                physicsClientId=self.env.pb_physics_client,
            )[0],
            dtype=float,
        )
        lifted_gripper_position = current_gripper_position.copy()
        lifted_gripper_position[2] += 0.1
        solution = p.calculateInverseKinematics(
            self.env.robot_id,
            self.gripper_link_index,
            targetPosition=lifted_gripper_position.tolist(),
            maxNumIterations=1000,
            residualThreshold=1e-4,
            lowerLimits=lower_limits.tolist(),
            upperLimits=upper_limits.tolist(),
            jointRanges=self.env.joint_ranges.tolist(),
            restPoses=current_qpos.tolist(),
            physicsClientId=self.env.pb_physics_client,
        )
        target_action = np.asarray(solution[: self.env.action_size], dtype=np.float32)
        target_action[-2:] = [-1, -1]
        self.target_action = target_action
        self.holding = True

    #################################### RETURN TO START ####################################
    def handle_return_to_start(self, obs):
        if self.returning_to_start == 0:
            self.target_action[0] = self.safe_qpos[0]
            self.returning_to_start = 1
            return

        if self.returning_to_start == 2:
            self.target_action[3:12] = self.safe_qpos[3:12]
            self.target_action[-2:] = [-1, -1]
            self.returning_to_start = 3
            return

        qvel = np.asarray(obs["qvel"])
        if np.all(np.abs(qvel) < self.movement_speed_threshold):
            print(f"return still step {self.still_steps}")
            self.still_steps += 1
        else:
            print(f"not still: {qvel}")
            self.still_steps = 0

        if self.still_steps >= self.still_steps_required:
            self.still_steps = 0

            if self.returning_to_start == 1:
                self.returning_to_start = 2
                return
            if self.returning_to_start == 3:
                self.state = GraspState.NAVIGATING
                self.returning_to_start = 0
        return

    #################################### NAVIGATION ####################################
    def _find_maze_entrance(self, cube_positions, goal_x):
        """Find the opening on the left boundary of the maze."""
        x_levels = np.unique(np.round(cube_positions[:, 0], 6))
        y_levels = np.unique(np.round(cube_positions[:, 1], 6))
        candidates = []

        for x in sorted(x_levels):
            if np.isclose(x, goal_x):
                continue

            wall_y = cube_positions[np.isclose(cube_positions[:, 0], x, atol=1e-6), 1]
            missing_y = [
                y for y in y_levels if not np.any(np.isclose(wall_y, y, atol=1e-6))
            ]
            if len(missing_y) == 1:
                candidates.append((float(x), float(missing_y[0])))

        if not candidates:
            return None

        return np.asarray(candidates[0], dtype=float)

    def _find_astar_path(self, start, goal, cube_positions):
        """Search the obstacle grid from the maze entrance to the goal."""
        x_levels = np.unique(np.round(cube_positions[:, 0], 6))
        y_levels = np.unique(np.round(cube_positions[:, 1], 6))

        if len(x_levels) == 0 or len(y_levels) == 0:
            return []

        def nearest_level(levels, position):
            return int(np.argmin(np.abs(levels - position)))

        def to_cell(position):
            x_index = nearest_level(x_levels, float(position[0]))
            y_index = nearest_level(y_levels, float(position[1]))
            if abs(x_levels[x_index] - float(position[0])) > 0.51:
                return None
            if abs(y_levels[y_index] - float(position[1])) > 0.51:
                return None
            return x_index, y_index

        start_cell = to_cell(start)
        goal_cell = to_cell(goal)
        if start_cell is None or goal_cell is None:
            return []

        blocked_cells = set()
        for x, y in cube_positions:
            x_index = nearest_level(x_levels, float(x))
            y_index = nearest_level(y_levels, float(y))
            blocked_cells.add((x_index, y_index))

        if start_cell in blocked_cells or goal_cell in blocked_cells:
            return []

        frontier = [
            (
                abs(start_cell[0] - goal_cell[0]) + abs(start_cell[1] - goal_cell[1]),
                0,
                start_cell,
            )
        ]
        came_from = {start_cell: None}
        cost_so_far = {start_cell: 0}

        while frontier:
            _, cost, current = heapq.heappop(frontier)
            if cost != cost_so_far[current]:
                continue
            if current == goal_cell:
                break

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (current[0] + dx, current[1] + dy)
                if (
                    neighbor[0] < 0
                    or neighbor[0] >= len(x_levels)
                    or neighbor[1] < 0
                    or neighbor[1] >= len(y_levels)
                    or neighbor in blocked_cells
                ):
                    continue

                new_cost = cost + 1
                if new_cost >= cost_so_far.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                cost_so_far[neighbor] = new_cost
                heuristic = abs(neighbor[0] - goal_cell[0]) + abs(
                    neighbor[1] - goal_cell[1]
                )
                heapq.heappush(
                    frontier,
                    (new_cost + heuristic, new_cost, neighbor),
                )

        if goal_cell not in came_from:
            return []

        path = []
        current = goal_cell
        while current is not None:
            path.append(
                np.asarray(
                    [x_levels[current[0]], y_levels[current[1]]],
                    dtype=float,
                )
            )
            current = came_from[current]
        path.reverse()
        return path

    def _simplify_navigation_path(self, path):
        """Keep only turn points and remove redundant points on straight segments."""
        if len(path) < 3:
            return [np.asarray(point, dtype=float) for point in path]

        simplified = [np.asarray(path[0], dtype=float)]
        for index in range(1, len(path) - 1):
            previous = np.asarray(path[index - 1], dtype=float)
            current = np.asarray(path[index], dtype=float)
            following = np.asarray(path[index + 1], dtype=float)

            previous_direction = np.sign(current - previous)
            following_direction = np.sign(following - current)
            if not np.array_equal(previous_direction, following_direction):
                simplified.append(current)

        simplified.append(np.asarray(path[-1], dtype=float))
        return simplified

    def _wrapped_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _follow_navigation_path(self, obs):
        robot_position = self.robot_root_position[:2] + np.asarray(
            obs["qpos"][:2],
            dtype=float,
        )

        while self.navigation_waypoint_index < len(self.navigation_waypoints):
            waypoint = np.asarray(
                self.navigation_waypoints[self.navigation_waypoint_index],
                dtype=float,
            )
            if (
                np.linalg.norm(waypoint - robot_position)
                > self.navigation_waypoint_tolerance
            ):
                break
            self.navigation_waypoint_index += 1

        if self.navigation_waypoint_index >= len(self.navigation_waypoints):
            print("Reached the navigation goal.")
            self.navigation_started = False
            self.state = GraspState.IDLE
            return

        waypoint = np.asarray(
            self.navigation_waypoints[self.navigation_waypoint_index],
            dtype=float,
        )
        target_offset = waypoint - self.robot_root_position[:2]
        heading = np.arctan2(
            waypoint[1] - robot_position[1],
            waypoint[0] - robot_position[0],
        )

        self.target_action[:2] = target_offset
        self.target_action[2] = self._wrapped_angle(heading)
        self.target_action[-2:] = self.action_lower_bound[-2:]

    def handle_navigation(self, obs):
        if not self.navigation_started:
            cube_positions = np.asarray(obs["cube_positions"], dtype=float)
            goal = np.asarray(obs["goal_pos"][:2], dtype=float)
            entrance = self._find_maze_entrance(cube_positions, goal[0])

            if entrance is None:
                print("Could not identify the maze entrance.")
                self.state = GraspState.FAILED
                return

            maze_path = self._find_astar_path(
                entrance,
                goal,
                cube_positions,
            )
            if not maze_path:
                print("Could not find an A* path from the entrance to the goal.")
                self.state = GraspState.FAILED
                return

            # Move to the entrance first, then follow the A* path through the maze.
            waypoints = [
                entrance,
                *maze_path,
            ]
            self.navigation_waypoints = self._simplify_navigation_path(waypoints)
            self.navigation_waypoint_index = 0
            self.navigation_started = True

        self._follow_navigation_path(obs)

    #################################### HELPERS ####################################
    def _get_state(self):
        agent_states = p.getJointStates(self.env.robot_id, self.env.joint_indices)
        qpos = np.array([s[0] for s in agent_states], dtype=np.float32)
        qvel = np.array([s[1] for s in agent_states], dtype=np.float32)

        object_pos, object_xyzw = p.getBasePositionAndOrientation(
            self.env.object_id, self.env.pb_physics_client
        )
        return qpos, qvel, np.array(object_pos), np.array(object_xyzw)

    def _get_object_obs(self):
        object_pos, object_xyzw = p.getBasePositionAndOrientation(self.env.object_id)
        object_rot = np.array(p.getMatrixFromQuaternion(object_xyzw)).reshape(3, 3)
        object_pc = (self.env.object_canonical_pc @ object_rot.T) + np.array(
            object_pos, dtype=np.float32
        ).reshape(1, 3)  # (1024, 3)
        object_normals = (
            self.env.object_canonical_normals @ object_rot.T
        )  # (1024, 3), outward normals
        return {
            "object_pc": object_pc,
            "object_normals": object_normals,
        }

    def _get_obs(self):
        qpos, qvel, object_pos, object_xyzw = (
            self._get_state()
        )  # （15，）（15,）（3,）（4，）
        object_obs = self._get_object_obs()
        # object_pc = object_obs["object_pc"]
        # object_normals = object_obs["object_normals"]
        # self.visualize_pc_and_normals(object_pc, object_normals, visualize_normals=False)
        return {
            "qpos": deepcopy(qpos),
            "qvel": deepcopy(qvel),
            "object_pos": deepcopy(object_pos),
            "object_xyzw": deepcopy(object_xyzw),
            "goal_pos": deepcopy(self.env.goal_pos),
            "object_pc": deepcopy(object_obs["object_pc"]),
            "object_normals": deepcopy(object_obs["object_normals"]),
            "cube_positions": deepcopy(self.env.cube_positions[:, :2]),
        }

    def evaluate(self):
        """
        Evaluate distance between the object position and the goal position.
        """
        _, _, object_pos, _ = self._get_state()
        dist_to_goal = np.linalg.norm(object_pos - self.env.goal_pos)
        success = dist_to_goal < 0.1

        return {
            "dist_to_goal": dist_to_goal,
            "success": success,
        }

    def visualize_point(self, x, y, z=0.5):
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=self.marker_visual,
            basePosition=[x, y, z],
        )

    ####################################  CORE LOGIC ####################################
    def handle_state(self, obs):
        if self.state == GraspState.WAITING:
            self.handle_waiting(obs)
        if self.state == GraspState.CANDIDATES:
            self.handle_candidate_generation(obs)
        if self.state == GraspState.SOLVE_IK:
            self.handle_solve_ik()

        if self.state == GraspState.ADJUSTING_GRIPPER:
            self.handle_gripper_adjustment(obs)

        if self.state == GraspState.FINAL_APPROACH:
            self.handle_final_approach(obs)

        if self.state == GraspState.APPROACHING:
            self.handle_approaching(obs)
        if self.state == GraspState.CLOSING:
            self.handle_closing(obs)
        if self.state == GraspState.HOLDING:
            self.handle_holding(obs)
        if self.state == GraspState.RETURN_TO_START:
            self.handle_return_to_start(obs)

        if self.state == GraspState.NAVIGATING:
            self.handle_navigation(obs)

    def get_action(self):
        obs = self._get_obs()
        # info = self.evaluate()

        self.handle_state(obs)

        delta = self.target_action - self.action
        self.action += np.clip(delta, -self.action_scale, self.action_scale)

        return np.clip(self.action, self.action_lower_bound, self.action_upper_bound)
