import pybullet as p
import numpy as np
import heapq
import time

from enum import Enum, auto
from copy import deepcopy

class GraspState(Enum):
    IDLE = auto()
    PREGRASPING = auto()
    APPROACHING = auto()
    FINAL_APPROACH = auto()
    CLOSING = auto()
    HOLDING = auto()
    FAILED = auto()
    TESTING = auto()
    
    CANDIDATES = auto()
    SOLVE_IK = auto()
    ADJUSTING_GRIPPER = auto()
    
    # FINAL ATTEMPT STATES
    WAITING = auto()
    NAVIGATING = auto()
    RETURN_TO_START = auto()
    
    ALERTING = auto()

class KeyBoardController:
    def __init__(self, env):
        ### GENERAL
        self.env = env
        self.action = env.agent_qpos # np.zeros(self.env.action_size)
        self.action_scale = np.array([0.03] * env.action_size) 
        self.action_scale[:3] = 0.02 # root movement
        self.action_scale[-2:] = 0.005  # gripper
        self.action_upper_bound = env.joint_upper
        self.action_lower_bound = env.joint_lower
        self.target_action = env.agent_qpos.copy()
        self.state = GraspState.WAITING
        
        self.gripper_link_index = next(
            j for j in range(p.getNumJoints(self.env.robot_id))
            if p.getJointInfo(self.env.robot_id, j)[12].decode() == "gripper_link"
        ) # index 20
        self.right_gripper_link_index = next(
            j for j in range(p.getNumJoints(self.env.robot_id))
            if p.getJointInfo(self.env.robot_id, j)[12].decode()
            == "r_gripper_finger_link"
        )
        self.left_gripper_link_index = next(
            j for j in range(p.getNumJoints(self.env.robot_id))
            if p.getJointInfo(self.env.robot_id, j)[12].decode()
            == "l_gripper_finger_link"
        )
        wrist_roll_link_index = next(
            j for j in range(p.getNumJoints(self.env.robot_id))
            if p.getJointInfo(self.env.robot_id, j)[12].decode() == "wrist_roll_link"
        )
        
       
        
        ### WAITING
        self.waiting_steps = 0
        self.curr_obj_pos = None
        
        ### GENERATING CANDIDATES
        self.grasp_candidates = []
        self.final_approach_action = None
        self.final_approach_still_steps = 0
        self.start_closing = False
        
        ### SOLVING IK
        self.max_top_to_midpoint_distance = 0.05
        self.target_position = []
        # self.target_midpoint = ()
        
        self.still_steps = 0
        self.still_steps_required = 3
        
        self.movement_speed_threshold = 0.3
        self.target_position_tolerance = 0.02
        
        
        
        
        self.waiting_steps_required = 5
        self.object_movement_threshold = 1e-4
        
        self.waypoints = []
        self.entrance_approach_waypoints = []
        self.navigation_waypoints = []
        self.navigation_waypoint_index = 0
        self.navigation_started = False
        self.robot_root_position = np.asarray(
            p.getBasePositionAndOrientation(
                self.env.robot_id,
                self.env.pb_physics_client,
            )[0],
            dtype=float,
        )
        self.navigation_waypoint_tolerance = 0.12
        self.navigation_front_clearance = 0.5
        self.final_approach = False
        self.marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0.1, 1.0, 0.1, 0.9],)
        self.test_flag = False
        
        self.closing_finger_qpos = None
        self.closing_stable_steps = 0
        self.closing_stable_steps_required = 5
        self.closing_stability_threshold = 1e-4
        
        self.holding = False
        self.returning_to_start = False
        
        self.safe_qpos = [ 7.8458279e-06,  8.6499352e-07,  9.3679247e-08, -9.2995993e-05,
            -8.7780473e-31,  4.6811578e-16, -1.1504074e-05,  1.5180924e+00,
            1.1136840e-04, -2.2503049e+00,  6.2935712e-04, -8.3095497e-01,
            1.5788910e-04, -1.5303650e-04,  2.1410917e-04
        ]
        self.ideal_grasp_qpos = [-1.73500027e-07,  6.73842067e-07, -2.71663474e-08, -1.13534967e-04,
            2.69898787e-28, -3.55052012e-12, -2.05587730e-06, -4.59455371e-01,
            -2.49372190e-03,  1.44456765e-02,  2.51946901e-03,  2.00252628e+00,
            -4.67113859e-05,  0.05, 0.05,
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
        pc, normals = obs['object_pc'], obs['object_normals']
        candidates = self.get_all_candidates(pc, normals)
        if not candidates:
            candidates = self.get_all_candidates(pc, normals, normal_threshold=0.8)
        
        if candidates:
            # self.visualize_grasp_candidates(candidates, length=1)
            self.grasp_candidates = candidates
            self.state = GraspState.SOLVE_IK
            
            self.final_approach_action = None
            self.final_approach_still_steps = 0
            
            self.start_closing = False
            self.closing_finger_qpos = None
            self.closing_stable_steps = 0
            

    def get_all_candidates(self, point_cloud, normals, normal_threshold = 0.95):
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
                
                candidates.append({
                    "indices": (i, j),
                    "points": (p1, p2),
                    "normals": (n1, n2),
                    "width": width,
                    "quality": quality,
                    "center_distance": center_distance,
                    "top_to_midpoint_distance": top_to_midpoint_distance,
                })

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
        solution = self.solve_ik() #is a list/
        if solution:
            self.target_action = np.asarray(solution, dtype=np.float32)
            self.target_action[-2:] = [1, 1]
            self.target_action = np.clip(
                        self.target_action,
                        self.action_lower_bound,
                        self.action_upper_bound,
                    )
            self.state = GraspState.APPROACHING


    def get_orientation(self, candidate):
        candidate_direction = (
            np.asarray(candidate["points"][0], dtype=float)
            - np.asarray(candidate["points"][1], dtype=float)
        )
        candidate_direction /= np.linalg.norm(candidate_direction)

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
        current_direction = np.asarray(left_state[4]) - np.asarray(right_state[4])
        current_direction /= np.linalg.norm(current_direction)

        if np.dot(candidate_direction, current_direction) < 0:
            candidate_direction = -candidate_direction

        local_y = candidate_direction
        local_z = np.array([0.0, 0.0, 1.0])
        local_z -= np.dot(local_z, local_y) * local_y
        if np.linalg.norm(local_z) < 1e-6:
            local_z = np.array([1.0, 0.0, 0.0])
            local_z -= np.dot(local_z, local_y) * local_y
        local_z /= np.linalg.norm(local_z)
        local_x = np.cross(local_y, local_z)
        local_x /= np.linalg.norm(local_x)

        rotation = np.column_stack((local_x, local_y, local_z))
        trace = np.trace(rotation)
        if trace > 0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    0.25 * scale,
                ]
            )
        else:
            diagonal = np.diag(rotation)
            index = int(np.argmax(diagonal))
            if index == 0:
                scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
                quaternion = np.array(
                    [
                        0.25 * scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[2, 1] - rotation[1, 2]) / scale,
                    ]
                )
            elif index == 1:
                scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
                quaternion = np.array(
                    [
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        0.25 * scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        (rotation[0, 2] - rotation[2, 0]) / scale,
                    ]
                )
            else:
                scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
                quaternion = np.array(
                    [
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        0.25 * scale,
                        (rotation[1, 0] - rotation[0, 1]) / scale,
                    ]
                )

        return (quaternion / np.linalg.norm(quaternion)).tolist()


    def get_orientation_v2(self, candidate):
        top_to_midpoint_distance = candidate["top_to_midpoint_distance"]
        verticle_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
        if top_to_midpoint_distance > self.max_top_to_midpoint_distance:
            print(
                "Grasp midpoint is too far below the object top: "
                f"{top_to_midpoint_distance:.3f} m"
            )
            return verticle_orn # TODO get an angle perpendicular to the grasp
        
        return verticle_orn


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

        target_rotation = np.asarray(
            p.getMatrixFromQuaternion(target_orn),
            dtype=float,
        ).reshape(3, 3)
        approach_direction = target_rotation[:, 0]
        # approach_marker_position = (
        #     np.asarray(self.target_position, dtype=float)
        #     + 0.05 * approach_direction
        # )
        # self.approach_marker_position = approach_marker_position

        current_qpos = self.env.agent_qpos
        lower_limits = self.action_lower_bound.copy()
        upper_limits = self.action_upper_bound.copy()
        # lower_limits[:3] = current_qpos[:3]
        # upper_limits[:3] = current_qpos[:3]
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
            restPoses=current_qpos.tolist(),
        )
        self.pre_approach_link_position = new_target_position
        # self.visualize_point(self.target_position[0], self.target_position[1], self.target_position[2])
        # self.visualize_point(new_target_position[0], new_target_position[1], new_target_position[2])
        
        solution = solution[:self.env.action_size]
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

        # Wait for the wrist to finish its current adjustment before measuring again.
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
            j for j in range(p.getNumJoints(robot_id))
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
        candidate_direction = (
            np.asarray(candidate["points"][0], dtype=float)
            - np.asarray(candidate["points"][1], dtype=float)
        )
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
    def handle_final_approach_original(self, obs):
        if self.final_approach_action is None:
            candidate = self.grasp_candidates[0]
            midpoint = (
                np.asarray(candidate["points"][0], dtype=float)
                + np.asarray(candidate["points"][1], dtype=float)
            ) / 2
            current_qpos = np.asarray(obs["qpos"], dtype=float)
            lower_limits = self.action_lower_bound.copy()
            upper_limits = self.action_upper_bound.copy()
            lower_limits[:3] = current_qpos[:3]
            upper_limits[:3] = current_qpos[:3]

            # solution = p.calculateInverseKinematics(
            #     self.env.robot_id,
            #     self.gripper_link_index,
            #     targetPosition=self.target_position,
            #     # maxNumIterations=1000,
            #     residualThreshold=1e-4,
            #     lowerLimits=lower_limits.tolist(),
            #     upperLimits=upper_limits.tolist(),
            #     jointRanges=self.env.joint_ranges.tolist(),
            #     restPoses=current_qpos.tolist(),
            #     physicsClientId=self.env.pb_physics_client,
            # # )
            # if not solution:
            #     print("Could not solve IK for the final grasp approach.")
            #     self.state = GraspState.FAILED
            #     return

            self.final_approach_action = np.asarray(
                solution[:self.env.action_size],
                dtype=np.float32,
            )
            self.final_approach_action[-2:] = self.action_upper_bound[-2:]
            self.final_approach_action = np.clip(
                self.final_approach_action,
                self.action_lower_bound,
                self.action_upper_bound,
            )

        self.target_action = self.final_approach_action.copy()
        qpos = np.asarray(obs["qpos"], dtype=float)
        qvel = np.asarray(obs["qvel"], dtype=float)
        error = self.target_action - qpos
        wrap_mask = np.isclose(self.action_upper_bound, 314.0)
        error[wrap_mask] = (error[wrap_mask] + np.pi) % (2 * np.pi) - np.pi

        if (
            np.all(np.abs(error) <= self.target_position_tolerance)
            and np.all(np.abs(qvel) < self.movement_speed_threshold)
        ):
            self.final_approach_still_steps += 1
        else:
            self.final_approach_still_steps = 0

        if self.final_approach_still_steps >= self.still_steps_required:
            print("At grasp pose; closing gripper.")
            self.state = GraspState.CLOSING


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
                self.state= GraspState.CLOSING
            return
        
        current_qpos = np.asarray(obs["qpos"], dtype=float)
        lower_limits = self.action_lower_bound.copy()
        upper_limits = self.action_upper_bound.copy()
        midpoint = self.target_midpoint
        midpoint[-1] += 0.02  # raise the target position slightly above the midpoint
        # self.visualize_point(midpoint[0], midpoint[1], midpoint[2])
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
        
        self.target_action = np.asarray(solution[:self.env.action_size], dtype=np.float32)
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
                self.state= GraspState.RETURN_TO_START
            return
            
        print("Start Holding object.")
        # Here you can implement any logic for holding the object, such as lifting it or moving it to a new location.
        # For now, we'll just keep the gripper closed and maintain the current position.
        self.target_action[-2:] = self.action_lower_bound[-2:]  # Keep gripper closed
        # You can add more logic here if needed.
        current_qpos = np.asarray(obs["qpos"], dtype=float)
        lower_limits = self.action_lower_bound.copy()
        upper_limits = self.action_upper_bound.copy()
        solution = p.calculateInverseKinematics(
                self.env.robot_id,
                self.gripper_link_index,
                targetPosition=self.pre_approach_link_position,
                maxNumIterations=1000,
                residualThreshold=1e-4,
                lowerLimits=lower_limits.tolist(),
                upperLimits=upper_limits.tolist(),
                jointRanges=self.env.joint_ranges.tolist(),
                restPoses=current_qpos.tolist(),
                physicsClientId=self.env.pb_physics_client,
        )
        target_action = np.asarray(solution[:self.env.action_size], dtype=np.float32)
        gripper_squeeze_offset = 0.003
        # target_action[-2:] = np.maximum(
        #     self.action_lower_bound[-2:],
        #     current_qpos[-2:] - gripper_squeeze_offset,
        # )
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
                self.state= GraspState.NAVIGATING
                self.returning_to_start = 0
        return
        
        
#################################### ADJUSTING ARM ####################################
    def draw_target_locations(self, target_orientation, target_gripper_position):
        robot_id = self.env.robot_id
        wrist_index = self.env.joint_indices[-3]

        wrist_state = p.getLinkState(
            robot_id,
            wrist_index,
            computeForwardKinematics=True,
        )

        wrist_position = np.asarray(wrist_state[4], dtype=float)
        target_wrist_rotation = np.asarray(
            p.getMatrixFromQuaternion(target_orientation),
            dtype=float,
        ).reshape(3, 3)
        closed_gripper_offset = np.array([0.16645, 0.015425, 0.0])
        target_gripper_position = np.asarray(
            target_gripper_position,
            dtype=float,
        )
        target_wrist_position = (
            target_gripper_position
            - target_wrist_rotation @ closed_gripper_offset
        )

        # marker_visual = p.createVisualShape(
        #     p.GEOM_SPHERE,
        #     radius=0.05,
        #     rgbaColor=[0.1, 1.0, 0.1, 0.9],
        # )
        for target_position in (target_wrist_position, target_gripper_position):
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=marker_visual,
                basePosition=target_position,
            )

        return target_wrist_position


    def test_adjusting(self,obs):
        robot_id = self.env.robot_id
        gripper_index = self.env.joint_indices[-1]
        
        gripper_link_state = p.getLinkState(
            robot_id,
            self.gripper_link_index,
            computeForwardKinematics=True,
        )
        left_state = p.getLinkState(
            robot_id,
            gripper_index,
            computeForwardKinematics=True,
        )

        gripper_link_position = np.asarray(gripper_link_state[4], dtype=float)
        gripper_position = np.asarray(left_state[4], dtype=float)

        # print(f"wrist: [{', '.join(f'{value:.4f}' for value in gripper_link_position)}]")
        # print(f"gripper: [{', '.join(f'{value:.4f}' for value in gripper_position)}]")
        # difference = gripper_link_position - gripper_position
        # print(
        #     f"Difference: [{', '.join(f'{value:.4f}' for value in difference)}]"
        # )

        # gripper_target = np.array([1.0, 1.0, 1.0])
        # target_orientation = p.getQuaternionFromEuler(
        #     [0.0, np.deg2rad(45.0), 0.0]
        # )
        # target_wrist_position = self.draw_target_locations(
        #     target_orientation,
        #     gripper_target,
        # )
        # print(
        #     "Wrist target for gripper [1.0000, 1.0000, 1.0000] "
        #     f"at 45 degrees around Y: "
        #     f"[{', '.join(f'{value:.4f}' for value in target_wrist_position)}]"
        # )

        marker_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.025,
            rgbaColor=[1.0, 0.1, 0.1, 0.9],
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=marker_visual,
            basePosition=gripper_link_position,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=marker_visual,
            basePosition=gripper_position,
        )
        self.target_action[-1] = 1
        self.target_action[-2] = 1
        self.state = GraspState.IDLE


    def handle_adjusting(self, obs):
        if self.grasp_candidates is not None:
            candidate = self.grasp_candidates[0]
        midpoint = (candidate["points"][0] + candidate["points"][1]) / 2
        target_orientation = p.getQuaternionFromEuler(
                    [0.0, np.deg2rad(45.0), 0.0]
                )
        target_orn = self.get_orientation(candidate)
        if target_orn is None:
            self.state = GraspState.IDLE
            return
        target_wrist_location = self.draw_target_locations(target_orientation, midpoint)
        current_qpos = self.env.agent_qpos
        lower_limits = self.action_lower_bound.copy()
        upper_limits = self.action_upper_bound.copy()
        lower_limits[:3] = current_qpos[:3]
        upper_limits[:3] = current_qpos[:3]
        solution = p.calculateInverseKinematics(
                    self.env.robot_id,
                    self.gripper_link_index,
                    targetPosition=midpoint,
                    maxNumIterations=1000,
                    residualThreshold=1e-4,
                    targetOrientation=target_orn,
                    lowerLimits=lower_limits.tolist(),
                    upperLimits=upper_limits.tolist(),
                    jointRanges=self.env.joint_ranges.tolist(),
                    restPoses=current_qpos.tolist(),
                )
        if solution:
            self.target_action = np.asarray(solution, dtype=np.float32)
            self.target_action[-2:] = [1, 1]
            self.target_action = np.clip(
                        self.target_action,
                        self.action_lower_bound,
                        self.action_upper_bound,
                    )
        self.state = GraspState.IDLE


#################################### NAVIGATION ####################################
    def _find_maze_entrance(self, cube_positions, goal_x):
        y_levels = np.unique(cube_positions[:, 1])
        entrance_candidates = []

        for x in np.unique(cube_positions[:, 0]):
            if np.isclose(x, goal_x):
                continue

            wall_y = cube_positions[np.isclose(cube_positions[:, 0], x), 1]
            missing_y = [
                y for y in y_levels
                if not np.any(np.isclose(wall_y, y))
            ]
            if len(missing_y) == 1:
                entrance_candidates.append((x, missing_y[0]))

        if entrance_candidates:
            return np.asarray(min(entrance_candidates), dtype=float)
        return None


    def _find_astar_path(self, start, goal, cube_positions):
        x_levels = np.unique(np.round(cube_positions[:, 0], 6))
        y_levels = np.unique(np.round(cube_positions[:, 1], 6))
        x_indices = {float(x): index for index, x in enumerate(x_levels)}
        y_indices = {float(y): index for index, y in enumerate(y_levels)}

        def to_cell(position):
            x, y = (round(float(value), 6) for value in position)
            if x not in x_indices or y not in y_indices:
                raise ValueError("A* start and goal must align with the maze grid.")
            return x_indices[x], y_indices[y]

        wall_positions = np.asarray(cube_positions, dtype=float)
        cell_width = float(np.min(np.diff(x_levels)))
        cell_height = float(np.min(np.diff(y_levels)))
        base_position = np.asarray(
            p.getBasePositionAndOrientation(
                self.env.robot_id,
                self.env.pb_physics_client,
            )[0],
            dtype=float,
        )
        base_aabbs = [
            p.getAABB(
                self.env.robot_id,
                -1,
                physicsClientId=self.env.pb_physics_client,
            )
        ]
        for link_index in range(p.getNumJoints(self.env.robot_id)):
            link_name = p.getJointInfo(
                self.env.robot_id,
                link_index,
                physicsClientId=self.env.pb_physics_client,
            )[12].decode()
            if link_name == "base_link" or "wheel" in link_name:
                base_aabbs.append(
                    p.getAABB(
                        self.env.robot_id,
                        link_index,
                        physicsClientId=self.env.pb_physics_client,
                    )
                )
        aabb_min = np.min([bounds[0] for bounds in base_aabbs], axis=0)
        aabb_max = np.max([bounds[1] for bounds in base_aabbs], axis=0)
        base_half_width_x = max(
            abs(aabb_min[0] - base_position[0]),
            abs(aabb_max[0] - base_position[0]),
        )
        base_half_width_y = max(
            abs(aabb_min[1] - base_position[1]),
            abs(aabb_max[1] - base_position[1]),
        )
        base_radius = np.hypot(base_half_width_x, base_half_width_y) + 0.05
        clearance_x_back = cell_width / 2 + base_radius
        clearance_x_front = clearance_x_back + self.navigation_front_clearance
        clearance_y = cell_height / 2 + base_radius

        def segment_is_clear(first_cell, second_cell):
            first = (x_levels[first_cell[0]], y_levels[first_cell[1]])
            second = (x_levels[second_cell[0]], y_levels[second_cell[1]])
            min_x = min(first[0], second[0]) - clearance_x_back
            max_x = max(first[0], second[0]) + clearance_x_front
            min_y = min(first[1], second[1]) - clearance_y
            max_y = max(first[1], second[1]) + clearance_y
            overlaps_walls = (
                (wall_positions[:, 0] >= min_x)
                & (wall_positions[:, 0] <= max_x)
                & (wall_positions[:, 1] >= min_y)
                & (wall_positions[:, 1] <= max_y)
            )
            return not np.any(overlaps_walls)

        start_cell = to_cell(start)
        goal_cell = to_cell(goal)
        if not segment_is_clear(start_cell, start_cell) or not segment_is_clear(
            goal_cell, goal_cell
        ):
            return []

        frontier = [(0, 0, start_cell)]
        came_from = {start_cell: None}
        cost_so_far = {start_cell: 0}

        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current_cost != cost_so_far[current]:
                continue
            if current == goal_cell:
                break

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = current[0] + dx, current[1] + dy
                if not (
                    0 <= neighbor[0] < len(x_levels)
                    and 0 <= neighbor[1] < len(y_levels)
                ) or not segment_is_clear(current, neighbor):
                    continue

                new_cost = current_cost + 1
                if new_cost >= cost_so_far.get(neighbor, float("inf")):
                    continue

                heuristic = (
                    abs(goal_cell[0] - neighbor[0])
                    + abs(goal_cell[1] - neighbor[1])
                )
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(
                    frontier,
                    (new_cost + heuristic, new_cost, neighbor),
                )

        if goal_cell not in came_from:
            return []

        path = []
        current = goal_cell
        while current is not None:
            path.append([
                float(x_levels[current[0]]),
                float(y_levels[current[1]]),
            ])
            current = came_from[current]
        return path[::-1]


    def _find_entrance_approach_path(self, start, goal, cube_positions):
        resolution = 0.25
        robot_position = np.asarray(start, dtype=float)
        base_aabbs = [
            p.getAABB(
                self.env.robot_id,
                -1,
                physicsClientId=self.env.pb_physics_client,
            )
        ]
        for link_index in range(p.getNumJoints(self.env.robot_id)):
            link_name = p.getJointInfo(
                self.env.robot_id,
                link_index,
                physicsClientId=self.env.pb_physics_client,
            )[12].decode()
            if link_name == "base_link" or "wheel" in link_name:
                base_aabbs.append(
                    p.getAABB(
                        self.env.robot_id,
                        link_index,
                        physicsClientId=self.env.pb_physics_client,
                    )
                )

        base_half_width_x = max(
            abs(bounds[0][0] - robot_position[0])
            for bounds in base_aabbs
        )
        base_half_width_x = max(
            base_half_width_x,
            max(bounds[1][0] - robot_position[0] for bounds in base_aabbs),
        )
        base_half_width_y = max(
            abs(bounds[0][1] - robot_position[1])
            for bounds in base_aabbs
        )
        base_half_width_y = max(
            base_half_width_y,
            max(bounds[1][1] - robot_position[1] for bounds in base_aabbs),
        )
        clearance = np.hypot(base_half_width_x, base_half_width_y) + 0.05
        clearance_front = clearance + self.navigation_front_clearance

        wall_positions = np.asarray(cube_positions, dtype=float)
        wall_x_levels = np.unique(np.round(wall_positions[:, 0], 6))
        wall_y_levels = np.unique(np.round(wall_positions[:, 1], 6))
        wall_half_width = min(
            float(np.min(np.diff(wall_x_levels))),
            float(np.min(np.diff(wall_y_levels))),
        ) / 2
        obstacles = [
            (
                x - wall_half_width - clearance,
                x + wall_half_width + clearance_front,
                y - wall_half_width - clearance,
                y + wall_half_width + clearance,
            )
            for x, y in wall_positions
        ]

        table_link_indices = [-1] + list(
            range(p.getNumJoints(
                self.env.table_id,
                physicsClientId=self.env.pb_physics_client,
            ))
        )
        for link_index in table_link_indices:
            bounds = p.getAABB(
                self.env.table_id,
                link_index,
                physicsClientId=self.env.pb_physics_client,
            )
            obstacles.append((
                bounds[0][0] - clearance,
                bounds[1][0] + clearance_front,
                bounds[0][1] - clearance,
                bounds[1][1] + clearance,
            ))

        origin = robot_position
        goal = np.asarray(goal, dtype=float)
        goal_cell = tuple(np.rint((goal - origin) / resolution).astype(int))
        min_x = int(np.floor((min(origin[0], goal[0]) - 2.0 - origin[0]) / resolution))
        max_x = int(np.ceil((max(origin[0], goal[0]) + 2.0 - origin[0]) / resolution))
        min_y = int(np.floor((min(origin[1], goal[1]) - 2.0 - origin[1]) / resolution))
        max_y = int(np.ceil((max(origin[1], goal[1]) + 2.0 - origin[1]) / resolution))

        def cell_position(cell):
            return origin + np.asarray(cell, dtype=float) * resolution

        def segment_is_clear(first_position, second_position):
            min_x_segment = min(first_position[0], second_position[0])
            max_x_segment = max(first_position[0], second_position[0])
            min_y_segment = min(first_position[1], second_position[1])
            max_y_segment = max(first_position[1], second_position[1])
            return not any(
                min_x_segment <= obstacle[1]
                and max_x_segment >= obstacle[0]
                and min_y_segment <= obstacle[3]
                and max_y_segment >= obstacle[2]
                for obstacle in obstacles
            )

        def is_clear(first, second):
            return segment_is_clear(cell_position(first), cell_position(second))

        start_cell = (0, 0)
        if not is_clear(start_cell, start_cell) or not segment_is_clear(
            cell_position(goal_cell),
            goal,
        ):
            return []

        frontier = [(0, 0, start_cell)]
        came_from = {start_cell: None}
        cost_so_far = {start_cell: 0}
        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current_cost != cost_so_far[current]:
                continue
            if current == goal_cell:
                break

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = current[0] + dx, current[1] + dy
                if not (
                    min_x <= neighbor[0] <= max_x
                    and min_y <= neighbor[1] <= max_y
                    and is_clear(current, neighbor)
                ):
                    continue
                new_cost = current_cost + 1
                if new_cost >= cost_so_far.get(neighbor, float("inf")):
                    continue
                heuristic = abs(goal_cell[0] - neighbor[0]) + abs(goal_cell[1] - neighbor[1])
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(
                    frontier,
                    (new_cost + heuristic, new_cost, neighbor),
                )

        if goal_cell not in came_from:
            return []

        path = []
        current = goal_cell
        while current is not None:
            path.append(cell_position(current).tolist())
            current = came_from[current]
        path.reverse()
        if not np.allclose(path[-1], goal):
            if not segment_is_clear(path[-1], goal):
                return []
            path.append(goal.tolist())
        return path


    def _follow_navigation_path(self, obs):
        if self.navigation_waypoint_index >= len(self.navigation_waypoints):
            self.navigation_started = False
            self.state = GraspState.IDLE
            print("Reached the maze goal.")
            return

        waypoint = np.asarray(
            self.navigation_waypoints[self.navigation_waypoint_index],
            dtype=float,
        )
        robot_position = self.robot_root_position[:2] + np.asarray(
            obs["qpos"][:2],
            dtype=float,
        )
        if np.linalg.norm(waypoint - robot_position) <= self.navigation_waypoint_tolerance:
            self.navigation_waypoint_index += 1
            if self.navigation_waypoint_index >= len(self.navigation_waypoints):
                self.navigation_started = False
                self.state = GraspState.IDLE
                print("Reached the maze goal.")
                return
            waypoint = np.asarray(
                self.navigation_waypoints[self.navigation_waypoint_index],
                dtype=float,
            )

        self.target_action[:2] = waypoint - self.robot_root_position[:2]


    def handle_navigation(self, obs):
        if self.navigation_started:
            self._follow_navigation_path(obs)
            return

        cube_positions = np.asarray(obs["cube_positions"], dtype=float)
        goal = np.asarray(obs["goal_pos"][:2], dtype=float)
        entrance = self._find_maze_entrance(cube_positions, goal[0])
        if entrance is None:
            print("Could not identify the maze entrance from cube_positions.")
            self.waypoints = []
            self.state = GraspState.IDLE
            return

        self.waypoints = self._find_astar_path(entrance, goal, cube_positions)
        if self.waypoints:
            print(f"Waypoints to goal: {self.waypoints}")
        else:
            print("Could not find a path from the entrance to the goal.")
            self.state = GraspState.IDLE
            return

        robot_position = self.robot_root_position[:2] + np.asarray(
            obs["qpos"][:2],
            dtype=float,
        )
        self.entrance_approach_waypoints = self._find_entrance_approach_path(
            robot_position,
            entrance,
            cube_positions,
        )
        if not self.entrance_approach_waypoints:
            print("Could not find a collision-free path to the maze entrance.")
            self.state = GraspState.IDLE
            return

        self.maze_entrance = entrance
        self.navigation_waypoints = (
            self.entrance_approach_waypoints + self.waypoints[1:]
        )
        self.navigation_waypoint_index = 0
        self.navigation_started = True
        print(f"Path to maze entrance and goal: {self.navigation_waypoints}")
        for waypoint in self.navigation_waypoints:
            self.visualize_point(*waypoint)
        self._follow_navigation_path(obs)


#################################### TESTING ####################################
    def handle_test(self, obs):
        if self.test_flag == True:
            return
        
        gripper_link_state = p.getLinkState(
            self.env.robot_id,
            self.gripper_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.env.pb_physics_client,
        )
        marker_position = np.asarray(gripper_link_state[4], dtype=float)
        marker_position += np.array([0.1, 0.0, 0.1])

        current_qpos = self.env.agent_qpos
        lower_limits = self.action_lower_bound.copy()
        upper_limits = self.action_upper_bound.copy()
        lower_limits[:3] = current_qpos[:3]
        upper_limits[:3] = current_qpos[:3]

        solution = p.calculateInverseKinematics(
            self.env.robot_id,
            self.gripper_link_index,
            targetPosition=marker_position,
            residualThreshold=1e-4,
            lowerLimits=lower_limits.tolist(),
            upperLimits=upper_limits.tolist(),
            jointRanges=self.env.joint_ranges.tolist(),
            restPoses=current_qpos.tolist(),
        )
        self.target_action = np.asarray(
            solution[:self.env.action_size], dtype=np.float32
        )
        self.target_action[:3] = current_qpos[:3]
        self.gripper_debug_marker_id = self.visualize_point(*marker_position)
        self.test_flag = True
        

#################################### HELPERS ####################################
    def _get_state(self):
        agent_states = p.getJointStates(self.env.robot_id, self.env.joint_indices)
        qpos = np.array([s[0] for s in agent_states], dtype=np.float32)
        qvel = np.array([s[1] for s in agent_states], dtype=np.float32)
        
        object_pos, object_xyzw = p.getBasePositionAndOrientation(self.env.object_id, self.env.pb_physics_client)
        return qpos, qvel, np.array(object_pos), np.array(object_xyzw)
    
    
    def _get_object_obs(self):
        object_pos, object_xyzw = p.getBasePositionAndOrientation(self.env.object_id)
        object_rot = np.array(p.getMatrixFromQuaternion(object_xyzw)).reshape(3, 3) 
        object_pc = (self.env.object_canonical_pc @ object_rot.T) + np.array(object_pos, dtype=np.float32).reshape(1, 3)  # (1024, 3)
        object_normals = (self.env.object_canonical_normals @ object_rot.T)  # (1024, 3), outward normals
        return {
            "object_pc": object_pc,
            "object_normals": object_normals,
        }
        
        
    def _get_obs(self):
        qpos, qvel, object_pos, object_xyzw = self._get_state() # （15，）（15,）（3,）（4，）
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
    
    
    def visualize_point(self, x, y, z = 0.5):
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=self.marker_visual,
            basePosition=[x, y, z],
        )
        
        
####################################  CORE LOGIC #################################### 
    def set_state(self):
        keys = p.getKeyboardEvents()
        for k, v in keys.items():
            if (k == ord('t') and (v & p.KEY_WAS_TRIGGERED)):
                self.state = GraspState.TESTING
           
                
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
            
        if self.state == GraspState.TESTING:
            self.handle_test(obs)
        
        if self.state == GraspState.NAVIGATING:
            self.handle_navigation(obs)
        
        if self.state == GraspState.ALERTING:
            print("ALERTING")
            time.sleep(1)
            
            
    def get_action(self):
        obs = self._get_obs()
        # info = self.evaluate()
        
        self.handle_state(obs)
        
        delta = self.target_action - self.action
        self.action += np.clip(delta, -self.action_scale, self.action_scale)
        
        return np.clip(self.action, self.action_lower_bound, self.action_upper_bound)