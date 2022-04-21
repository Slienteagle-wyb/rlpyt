import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from collections import deque
import airsimdroneracingvae as airsim_vae
from airsimdroneracingvae.types import *
from airsimdroneracingvae.utils import *
from rlpyt.envs.base import Env
from rlpyt.spaces.int_box import IntBox
import racing_utils


class DroneGateEnv(Env):
    def __init__(self,
                 img_res: tuple,
                 action_repeat=4,
                 frame_stack=1,
                 gate_offset=(0, 0, -0),
                 vel_max=15.0,
                 acc_max=9.0,
                 linear_vel_scale=1.0,
                 yaw_vel_scale=1.2,
                 env_horizon=4000,
                 num_gates_track=8,
                 race_course_radius=8,
                 radius_noise=1.5,
                 direction=0
                 ):
        # gym style env config
        self._action_space = IntBox(low=-1, high=1, shape=(4, ))
        self._observation_space = IntBox(low=0, high=256,
                                         shape=(3 * frame_stack, img_res[0], img_res[1]),
                                         dtype=np.uint8)
        self._padding_obs = np.zeros((3, img_res[0], img_res[1]))
        if frame_stack > 1:
            self._obs_deque = deque(maxlen=frame_stack)
        self.env_horizon = env_horizon
        self.env_step_counter = 0
        self._frame_stack = frame_stack
        self._action_repeat = action_repeat
        # airsim scene config
        self.vel_max = vel_max
        self.acc_max = acc_max
        self.num_gates_track = num_gates_track
        self.race_course_radius = race_course_radius
        self.radius_noise = radius_noise
        self.direction = direction
        self._linear_vel_scale = linear_vel_scale
        self._yaw_vel_scale = yaw_vel_scale
        # gate track pointers
        self.next_gate_idx = 0
        self.last_passed_gate_idx = -1
        self.gate_passed_thresh = 0.5
        self.current_num_lap = 0
        self.success_fly_through = False
        # reward_shaping params
        self.d_max = 2.5  # hyper param of 2.5, 5.0, 0.0000001(means no safety penalty)
        self.w_g = 1.5
        self.safety_factor = 1.0
        self.penalty_factor = 0.5
        # load the level of airsim scene
        self.client = airsim_vae.MultirotorClient()
        self.img_client = airsim_vae.MultirotorClient()
        self.client.confirmConnection()
        self.img_client.confirmConnection()
        self.client.simLoadLevel('Soccer_Field_Easy')
        time.sleep(2.0)
        self.drone_name = 'drone_0'
        self.drone_current_state = None
        self.drone_last_state = None
        self.client.enableApiControl(True, vehicle_name=self.drone_name)
        time.sleep(0.01)
        self.client.armDisarm(True, vehicle_name=self.drone_name)
        time.sleep(0.01)
        self.client.setTrajectoryTrackerGains(airsim_vae.TrajectoryTrackerGains().to_list(),
                                              vehicle_name=self.drone_name)
        time.sleep(0.01)
        self.set_current_track_gate_poses_from_default_track_in_binary()
        time.sleep(1.0)
        takeoff_position = Vector3r(self.next_track_gate_poses[0].position.x_val - 1.5,
                                    self.next_track_gate_poses[0].position.y_val - 3.0,
                                    self.next_track_gate_poses[0].position.z_val)
        takeoff_orientation = Quaternionr(x_val=0.4, y_val=0.9, z_val=0, w_val=0)
        self.client.moveOnSplineVelConstraintsAsync([takeoff_position], [takeoff_orientation], vel_max=vel_max,
                                                    acc_max=acc_max, vehicle_name=self.drone_name,
                                                    viz_traj=False).join()
        time.sleep(3.0)

    def reset(self):
        for _ in range(self._frame_stack - 1):
            self._obs_deque.append(self._padding_obs)
        done = False
        obs = self.update_obs(done)
        self.drone_last_state = None
        self.drone_current_state = None
        last_passed_gate_pos = self.curr_track_gate_poses[self.last_passed_gate_idx].position.to_numpy_array()
        next_track_gate_pos = self.curr_track_gate_poses[self.next_gate_idx].position.to_numpy_array()
        interp_reset_pos = last_passed_gate_pos + (next_track_gate_pos - last_passed_gate_pos) * np.random.uniform(low=0.2, high=0.8)
        interp_reset_pos = interp_reset_pos.astype(np.float64)
        interp_reset_pos_airsim = Vector3r(interp_reset_pos[0], interp_reset_pos[1], interp_reset_pos[2])
        reset_quad_airsim = self.curr_track_gate_poses[self.next_gate_idx].orientation
        reset_quad = racing_utils.get_gate_facing_vector_from_quaternion(reset_quad_airsim)
        self.client.moveOnSplineVelConstraintsAsync([interp_reset_pos_airsim], [reset_quad], vel_max=self.vel_max,
                                                    acc_max=self.acc_max, vehicle_name=self.drone_name,
                                                    viz_traj=True).join()
        time.sleep(0.01)
        return obs

    def update_obs(self, done):
        obs = self._padding_obs if done else \
            self.img_client.simGetImages([airsim_vae.ImageRequest('0', airsim_vae.ImageType.Scene, False, False)])[0]
        if self._frame_stack > 1:
            self._obs_deque.append(obs)
            obs = np.concatenate(self._obs_deque)
        return obs

    def step(self, action):
        action[0:2] = action[0:2] * self._linear_vel_scale
        action[3] = action[3] * self._yaw_vel_scale
        yaw_mode = airsim_vae.YawMode(is_rate=True, yaw_or_rate=action[3] * 180.0 / np.pi)
        for _ in range(self._action_repeat):
            self.client.moveByVelocityAsync(action[0], action[1], action[2], duration=0.1,
                                            drivetrain=DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=yaw_mode)
        self.drone_last_state = self.drone_current_state
        self.drone_current_state = self.client.getMultirotorState()
        self.cam_info = self.img_client.simGetCameraInfo(camera_name='0', vehicle_name=self.drone_name)
        current_pos_np = self.drone_current_state.kinematics_estimated.position.to_numpy_array()
        if self.drone_last_state is None:
            last_pos_np = current_pos_np
        else:
            last_pos_np = self.drone_last_state.kinematics_estimated.position.to_numpy_array()
        next_gate_pos_np = self.curr_track_gate_poses[self.next_gate_idx].position.to_numpy_array()
        dist_from_next_gate = math.sqrt((current_pos_np[0] - next_gate_pos_np[0]) ** 2 +
                                        (current_pos_np[1] - next_gate_pos_np[1]) ** 2 +
                                        (current_pos_np[2] - next_gate_pos_np[2]) ** 2)
        if dist_from_next_gate < self.gate_passed_thresh:
            self.next_gate_idx += 1
            self.last_passed_gate_idx += 1
            self.set_pose_of_gate_before_last_passed()
            self.success_fly_through = True
            print('success fly through a gate !!!')


            if self.last_passed_gate_idx == len(self.curr_track_gate_poses) - 1:
                self.last_passed_gate_idx = -1
                self.next_gate_idx = 0
                self.curr_track_gate_poses = self.next_track_gate_poses
                self.next_track_gate_poses = self.get_next_generated_track()
                self.current_num_lap += 1
        self.env_step_counter += self._action_repeat
        done, next_gate_pos_camera = self.get_done(dist_from_next_gate, next_gate_pos_np)
        reward = self.get_reward(last_pos_np, current_pos_np, next_gate_pos_camera, done)
        obs = self.update_obs(done)
        return obs, reward, done, {}

    def get_done(self, dist_from_next_gate, next_gate_pos_np):
        col_info = self.img_client.simGetCollisionInfo(vehicle_name=self.drone_name)
        has_collided = col_info.has_collided
        camera_rot_matrix = Rotation.from_quat([self.cam_info.pose.orientation.x_val,
                                                self.cam_info.pose.orientation.y_val,
                                                self.cam_info.pose.orientation.z_val,
                                                self.cam_info.pose.orientation.w_val])
        camera_trans_vector = self.cam_info.pose.position.to_numpy_array()
        trans_matrix = np.empty((4, 4))
        trans_matrix[:3, :3] = camera_rot_matrix.as_matrix()
        trans_matrix[:3, 3] = camera_trans_vector
        trans_matrix[3, :] = [0., 0., 0., 1.0]
        w_to_came_trans_matrix = np.linalg.inv(trans_matrix)
        next_gate_pos_camera = np.matmul(w_to_came_trans_matrix, np.append(next_gate_pos_np, 1.0))
        next_gate_pos_center_fov = np.arctan(math.sqrt(next_gate_pos_camera[1] ** 2 + next_gate_pos_camera[2] ** 2) /
                                             next_gate_pos_camera[0]) * 180.0 / np.pi
        dist_to_center_axis = np.sqrt(next_gate_pos_camera[1] ** 2 + next_gate_pos_camera[2] ** 2)
        if has_collided or self.env_step_counter % self.env_horizon == 0:
            if has_collided:
                print('collision has been detected and num of step is:', self.env_step_counter)
            else:
                print('current num of env step {0} is out of horizon'.format(self.env_step_counter))
            done = True
        elif dist_from_next_gate > 10.0:
            done = True
            print('the dist bound is larger than 10.0, and num of step is:', self.env_step_counter)
        elif next_gate_pos_center_fov > 50.0 or next_gate_pos_center_fov < 0.0:
            if dist_from_next_gate < 0.75 or dist_to_center_axis < 1.0 or self.success_fly_through:
                done = False
                self.success_fly_through = False
            else:
                print('the next track gate is out of sight, fov is {0}, num of step is {1},\n \
                      dist_from_next_gate is {2}, dist to center axis is {3}'.format(next_gate_pos_center_fov,
                                                                                     self.env_step_counter,
                                                                                     dist_from_next_gate,
                                                                                     dist_to_center_axis))
                done = True
        else:
            done = False
        return done, next_gate_pos_camera

    def get_reward(self, last_pos_np, current_pos_np, next_gate_pos_camera, episode_terminal):
        if self.drone_last_state is None:
            r_progress = 0.
            r_safety = 0.
            r_terminal = 0.
            r_penalty = 0.
        else:
            last_passed_gate_pos = self.curr_track_gate_poses[self.last_passed_gate_idx].position.to_numpy_array()
            next_track_gate_pos = self.curr_track_gate_poses[self.next_gate_idx].position.to_numpy_array()
            # calculate the r_progress
            segment_vector = next_track_gate_pos - last_passed_gate_pos
            proj_vector = segment_vector / np.linalg.norm(segment_vector)
            progress_vector = current_pos_np - last_pos_np
            progress_proj = np.dot(progress_vector, proj_vector)
            r_progress = progress_proj
            # calculating the r_safety, when drone is close to center axis, f of r_safety is 0,
            # so the r_safety is also to be 0.
            d_p = np.abs(next_gate_pos_camera[0])  # the distance to gate vertical plane
            d_n = np.sqrt(next_gate_pos_camera[1] ** 2 + next_gate_pos_camera[2] ** 2)
            d_max = self.d_max
            w_g = self.w_g
            f = np.max([1.0 - (d_p / d_max), 0.0])
            v = np.max([(1 - f) * (w_g / 6.0), 0.05])
            r_safety = -(f**2) * (1.0 - np.exp(-0.5 * (d_n ** 2) / v))
            # calculate the r_terminal
            if episode_terminal:
                if self.env_step_counter % self.env_horizon == 0:
                    print('episode end w/o crash....')
                    r_terminal = 0.
                else:
                    d_g = np.sqrt(next_gate_pos_camera[0] ** 2 + next_gate_pos_camera[1] ** 2 +
                                  next_gate_pos_camera[2] ** 2)
                    r_terminal = -np.min([(d_g / w_g) ** 2, 20.0])
            else:
                r_terminal = 0
            # calculate the quadratic penalty
            # current_angular_rates = self.drone_current_state.kinematics_estimated.angular_acceleration.to_numpy_array()
            # r_penalty = -np.linalg.norm(current_angular_rates)
        total_reward = r_progress + self.safety_factor * r_safety + r_terminal
        print(total_reward, r_progress, r_safety, r_terminal)
        return total_reward

    def set_current_track_gate_poses_from_default_track_in_binary(self):
        gate_names_sorted_bad = sorted(self.client.simListSceneObjects("Gate.*"))
        gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
        gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k])
        self.gate_object_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]
        # limit the number of gates in the track
        self.gate_object_names_sorted = self.gate_object_names_sorted[:self.num_gates_track]

        self.curr_track_gate_poses = [self.client.simGetObjectPose(gate_name) for gate_name in
                                      self.gate_object_names_sorted]
        # destroy all previous gates in map
        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            self.client.simDestroyObject(gate_object)
            time.sleep(0.05)
        # generate track with correct number of gates
        self.next_track_gate_poses = racing_utils.generate_gate_poses(num_gates=self.num_gates_track,
                                                                      race_course_radius=self.race_course_radius,
                                                                      radius_noise=self.radius_noise,
                                                                      height_range=[0, -self.radius_noise],
                                                                      direction=self.direction,
                                                                      type_of_segment='circle')
        self.curr_track_gate_poses = self.next_track_gate_poses
        # create red gates in their places
        for idx in range(len(self.gate_object_names_sorted)):
            self.client.simSpawnObject(self.gate_object_names_sorted[idx], "RedGate16x16",
                                       self.next_track_gate_poses[idx], 0.75)
            time.sleep(0.05)

    def set_pose_of_gate_before_last_passed(self):
        gate_idx_to_move = self.last_passed_gate_idx - 1
        if self.last_passed_gate_idx in [-1, 0]:
            print("last_gate_passed_idx", self.last_passed_gate_idx, "moving gate idx from CURRENT track",
                  gate_idx_to_move)
            self.client.simSetObjectPose(self.gate_object_names_sorted[gate_idx_to_move],
                                         self.curr_track_gate_poses[gate_idx_to_move])
            return
        else:
            print("last_gate_passed_idx", self.last_passed_gate_idx, "moving gate idx from CURRENT track",
                  gate_idx_to_move)
            self.client.simSetObjectPose(self.gate_object_names_sorted[gate_idx_to_move],
                                         self.next_track_gate_poses[gate_idx_to_move])
            return

    def get_next_generated_track(self):
        return racing_utils.generate_gate_poses(num_gates=self.num_gates_track,
                                                race_course_radius=self.race_course_radius,
                                                radius_noise=self.radius_noise,
                                                height_range=[0, -self.radius_noise],
                                                direction=self.direction,
                                                type_of_segment='circle')


if __name__ == '__main__':
    drone_env = DroneGateEnv(img_res=(84, 84))
    for i in range(500):
        obs, reward, done, _ = drone_env.step(np.array([0.15, 1.0, 0.0, 0.0]))
        if done:
            drone_env.reset()
            time.sleep(2.0)
    time.sleep(5.0)
