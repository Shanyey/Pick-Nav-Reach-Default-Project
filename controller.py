import heapq  # noqa: F401
import time  # noqa: F401
from copy import deepcopy
from enum import Enum, auto

import numpy as np
import pybullet as p


class GraspState(Enum):
    WAITING = auto()
    CANDIDATES = auto()
    SOLVE_IK = auto()
    APPROACHING = auto()
    PREGRASP_READY = auto()
    FAILED = auto()
    NAVIGATING = auto()


class Controller:
    def __init__(self, env):
        self.env = env
        self.action_scale = np.array([0.03] * env.action_size)
        self.action_scale[:3] = 0.02  # root movement
        self.action_scale[-2:] = 0.005  # gripper
        self.action_upper_bound = env.joint_upper
        self.action_lower_bound = env.joint_lower
        self.marker_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.02, rgbaColor=[0.1, 1.0, 0.1, 0.9]
        )
        self.state = GraspState.WAITING
        self.target_action = env.agent_qpos.copy()
        self.action = env.agent_qpos.copy()
        self.approach_debug_ids = []

        ### WAITING
        self.waiting_steps = 0
        self.waiting_steps_required = 5
        self.curr_obj_pos = None
        self.object_movement_threshold = 1e-4

        ### APPROACHING
        self.gripper_link_index = next(
            j
            for j in range(p.getNumJoints(self.env.robot_id))
            if p.getJointInfo(self.env.robot_id, j)[12].decode() == "gripper_link"
        )
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

        self.approach_distance = 0.1
        # Use the gripper_link origin as the centred grasp reference. The old
        # +Y offset shifted this reference sideways along the closing axis.
        # If the model's contact-pad centre differs, calibrate this local offset
        # from its URDF/geometry; do not reuse a single finger's lateral offset.
        self.grasp_center_offset = np.zeros(3, dtype=float)
        self.pregrasp_position_tolerance = 0.005  # metres
        self.pregrasp_orientation_tolerance = np.deg2rad(5.0)
        self.pregrasp_debug_ids = []
        # Joint velocities use rad/s for revolute joints and m/s for prismatic joints.
        self.movement_speed_threshold = 0.02
        # Joint position errors use radians or meters, respectively.
        self.target_position_tolerance = 0.02
        self.still_steps = 0
        self.still_steps_required = 5

        ### NAVIGATION
        self.navigation_started = False
        self.navigation_waypoints = []
        self.navigation_waypoint_index = 0
        self.navigation_waypoint_tolerance = 0.12
        self.navigation_heading_tolerance = np.deg2rad(8.0)
        root_position, root_orientation = p.getBasePositionAndOrientation(
            self.env.robot_id,
            self.env.pb_physics_client,
        )
        self.robot_root_position = np.asarray(root_position, dtype=float)
        self.robot_root_yaw = float(p.getEulerFromQuaternion(root_orientation)[2])

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
            self.grasp_candidates = candidates
            for candidate in self.grasp_candidates:
                candidate["approach_direction"] = self.get_best_approach_direction(
                    candidate
                )
            self.visualize_best_approaches(candidates)
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

                # Skip if point is in the bottom half of the object
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

                # Skip if normals are not good enough
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
                        "midpoint": grasp_center,
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

    def get_best_approach_direction(self, candidate, side_clearance=0.05):
        p1 = np.asarray(candidate["points"][0], dtype=float)
        p2 = np.asarray(candidate["points"][1], dtype=float)
        midpoint = (p1 + p2) / 2.0

        # The line between the contact points is the gripper's closing axis.
        closing_axis = p2 - p1
        length = np.linalg.norm(closing_axis)
        if length < 1e-12:
            raise ValueError("Candidate points are too close together")
        closing_axis /= length

        # Create two unit vectors spanning the plane perpendicular to the closing axis.
        reference_axis = np.eye(3)[np.argmin(np.abs(closing_axis))]
        u = np.cross(closing_axis, reference_axis)
        u /= np.linalg.norm(u)

        v = np.cross(closing_axis, u)
        v /= np.linalg.norm(v)

        # Sample 360 possible approach directions, one per degree.
        angles = np.deg2rad(np.arange(360))
        directions = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v

        # Get the top of the table's world-space bounding box.
        table_link_indices = range(
            -1,
            p.getNumJoints(self.env.table_id),
        )
        table_top_z = max(
            p.getAABB(self.env.table_id, link_index)[1][2]
            for link_index in table_link_indices
        )

        height_above_table = midpoint[2] - table_top_z

        if height_above_table < side_clearance:
            # The grasp is low, so choose the most upward perpendicular direction.
            best_index = np.argmax(directions[:, 2])
            return directions[best_index]

        # Otherwise, prefer approaching horizontally from the robot's side.
        robot_base_position = np.asarray(
            p.getBasePositionAndOrientation(self.env.robot_id)[0],
            dtype=float,
        )
        preferred = robot_base_position - midpoint
        preferred[2] = 0.0

        # Keep the approach perpendicular to the closing axis.
        preferred -= np.dot(preferred, closing_axis) * closing_axis

        if np.linalg.norm(preferred) < 1e-6:
            # Fallback if the robot-side direction is parallel to the closing axis.
            horizontal_axes = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            )
            reference = horizontal_axes[
                np.argmin(np.abs(horizontal_axes @ closing_axis))
            ]
            preferred = reference - np.dot(reference, closing_axis) * closing_axis

        preferred /= np.linalg.norm(preferred)

        # Select the sampled direction closest to the preferred side approach.
        best_index = np.argmax(directions @ preferred)
        return directions[best_index]

    def visualize_best_approaches(self, candidates, line_length=None):
        if line_length is None:
            line_length = self.approach_distance
        # Remove lines from a previous candidate display.
        for debug_id in self.approach_debug_ids:
            p.removeUserDebugItem(debug_id)
        self.approach_debug_ids.clear()

        for index, candidate in enumerate(candidates):
            midpoint = np.asarray(candidate["midpoint"], dtype=float)
            if "approach_direction" not in candidate:
                candidate["approach_direction"] = self.get_best_approach_direction(
                    candidate
                )
            # Draw the same stored direction that solve_ik() will use.
            direction = np.asarray(candidate["approach_direction"], dtype=float)

            line_end = midpoint + line_length * direction

            line_id = p.addUserDebugLine(
                lineFromXYZ=midpoint.tolist(),
                lineToXYZ=line_end.tolist(),
                lineColorRGB=[0.0, 0.7, 1.0],
                lineWidth=2,
                lifeTime=0,
            )
            text_id = p.addUserDebugText(
                text=str(index + 1),
                textPosition=line_end.tolist(),
                textColorRGB=[0.0, 0.0, 0.0],
                textSize=1.2,
                lifeTime=0,
            )

            self.approach_debug_ids.extend([line_id, text_id])

    #################################### SOLVE IK ####################################
    def get_grasp_orientation(self, candidate):
        p1 = np.asarray(candidate["points"][0], dtype=float)
        p2 = np.asarray(candidate["points"][1], dtype=float)

        closing_axis = p2 - p1
        closing_axis /= np.linalg.norm(closing_axis)

        # Keep the closing-axis sign consistent with the current gripper.
        right_state = p.getLinkState(
            self.env.robot_id,
            self.right_gripper_link_index,
            computeForwardKinematics=True,
        )
        left_state = p.getLinkState(
            self.env.robot_id,
            self.left_gripper_link_index,
            computeForwardKinematics=True,
        )
        current_closing_axis = np.asarray(left_state[4], dtype=float) - np.asarray(
            right_state[4], dtype=float
        )
        current_closing_axis /= np.linalg.norm(current_closing_axis)

        if np.dot(closing_axis, current_closing_axis) < 0:
            closing_axis = -closing_axis

        # Stored direction points outward toward the pre-grasp position.
        # The gripper's local +X points inward, toward the grasp.
        approach_axis = -np.asarray(candidate["approach_direction"], dtype=float)
        approach_axis -= np.dot(approach_axis, closing_axis) * closing_axis
        approach_axis /= np.linalg.norm(approach_axis)

        # Desired local and world axes, as matrix columns.
        local_approach = np.array([1.0, 0.0, 0.0])
        local_closing = np.array([0.0, 1.0, 0.0])
        local_third = np.cross(local_approach, local_closing)

        world_third = np.cross(approach_axis, closing_axis)

        local_basis = np.column_stack((local_approach, local_closing, local_third))
        world_basis = np.column_stack((approach_axis, closing_axis, world_third))
        rotation = world_basis @ local_basis.T

        # Convert the rotation matrix to PyBullet's quaternion format.
        sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
        if sy > 1e-6:
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        else:
            roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
            yaw = 0.0

        pitch = np.arctan2(-rotation[2, 0], sy)
        return p.getQuaternionFromEuler([roll, pitch, yaw])

    def _gripper_link_target_position(self, contact_position, orientation):
        rotation = np.asarray(
            p.getMatrixFromQuaternion(orientation),
            dtype=float,
        ).reshape(3, 3)

        return np.asarray(contact_position, dtype=float) - (
            rotation @ self.grasp_center_offset
        )

    def solve_ik(self):
        if not self.grasp_candidates:
            return None

        # Start by trying the first candidate in your sorted top 10.
        candidate = self.grasp_candidates[0]
        midpoint = np.asarray(candidate["midpoint"], dtype=float)
        outward_direction = np.asarray(candidate["approach_direction"], dtype=float)
        # outward_direction /= np.linalg.norm(outward_direction)

        # loc = midpoint + 0.1 * outward_direction
        # This is the pre-grasp target, before moving toward the object.
        pregrasp_position = midpoint + self.approach_distance * outward_direction
        self.visualize_point(list=pregrasp_position.tolist())
        target_orientation = self.get_grasp_orientation(candidate)

        # target_link_position = self._gripper_link_target_position(
        #     pregrasp_position,
        #     target_orientation,
        # )

        # Keep the desired Cartesian pose so arrival can be checked independently
        # of whether the joints reached the IK solver's returned configuration.
        self.target_midpoint = midpoint.copy()
        self.target_approach_direction = outward_direction.copy()
        self.pregrasp_position = pregrasp_position.copy()
        # self.pregrasp_link_position = target_link_position.copy()
        self.pregrasp_orientation = np.asarray(target_orientation, dtype=float)
        # self.visualize_pregrasp_pose()

        current_qpos = np.asarray(self.env.agent_qpos, dtype=float)
        lower_limits = self.action_lower_bound.copy()
        upper_limits = self.action_upper_bound.copy()

        # Keep the mobile base at its current position during arm IK.
        lower_limits[:3] = current_qpos[:3]
        upper_limits[:3] = current_qpos[:3]

        solution = p.calculateInverseKinematics(
            bodyUniqueId=self.env.robot_id,
            endEffectorLinkIndex=self.gripper_link_index,
            # targetPosition=target_link_position.tolist(),
            targetPosition=pregrasp_position,
            targetOrientation=target_orientation,
            lowerLimits=lower_limits.tolist(),
            upperLimits=upper_limits.tolist(),
            jointRanges=self.env.joint_ranges.tolist(),
            # restPoses=current_qpos.tolist(),
            maxNumIterations=1000,
            residualThreshold=1e-4,
        )

        return np.asarray(
            solution[: self.env.action_size],
            dtype=np.float32,
        )

    def handle_solve_ik(self):
        solution = self.solve_ik()
        if solution is None:
            self.state = GraspState.WAITING
            return

        # Keep the gripper open while moving to the pre-grasp pose.
        solution[-2:] = self.action_upper_bound[-2:]
        self.target_action = np.clip(
            solution,
            self.action_lower_bound,
            self.action_upper_bound,
        )
        self.still_steps = 0
        self.state = GraspState.APPROACHING
        print("Approaching the pre-grasp pose with the gripper open.")

    #################################### APPROACHING ####################################
    def handle_approaching(self, obs):
        pass

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

        # The entrance is on the first vertical boundary after the left-room
        # walls, so choose the smallest x coordinate with exactly one gap.
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
            self.state = GraspState.PREGRASP_READY
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
            print(
                f"Navigation planned: entrance plus "
                f"{len(maze_path)} A* cells, "
                f"{len(self.navigation_waypoints)} waypoints."
            )

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

    def visualize_point(self, x=0, y=0, z=0.5, list=None):
        if list:
            return p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=self.marker_visual,
                basePosition=list,
            )
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=self.marker_visual,
            basePosition=[x, y, z],
        )

    def is_moving(self, obs, threshold=None):
        """Return whether any measured joint speed exceeds the threshold.

        This checks one observation; the caller counts consecutive still steps.
        Invalid velocities are treated as moving so they cannot indicate arrival.
        """
        if threshold is None:
            threshold = self.movement_speed_threshold
        qvel = np.asarray(obs["qvel"], dtype=float)
        if not np.all(np.isfinite(qvel)):
            return True
        return bool(np.any(np.abs(qvel) > threshold))

    def get_action(self):
        obs = self._get_obs()
        # info = self.evaluate()

        self.handle_state(obs)

        delta = self.target_action - self.action
        self.action += np.clip(delta, -self.action_scale, self.action_scale)

        return np.clip(self.action, self.action_lower_bound, self.action_upper_bound)

    def handle_state(self, obs):
        if self.state == GraspState.WAITING:
            self.handle_waiting(obs)
        elif self.state == GraspState.CANDIDATES:
            self.handle_candidate_generation(obs)
        elif self.state == GraspState.SOLVE_IK:
            self.handle_solve_ik()
        elif self.state == GraspState.APPROACHING:
            self.handle_approaching(obs)
        elif self.state == GraspState.NAVIGATING:
            self.handle_navigation(obs)
        # PREGRASP_READY and FAILED hold the existing target at this milestone.
