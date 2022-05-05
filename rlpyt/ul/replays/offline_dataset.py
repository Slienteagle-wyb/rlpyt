import numpy as np
import os
import cv2
import glob
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as T
from tqdm import tqdm
from torch.utils.data import Dataset
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_from_example

OfflineSamples = namedarraytuple('OfflineSamples', ['observation', 'translation', 'rotation', 'velocity', 'direction',
                                                    'attitude'])


class OfflineDatasets(Dataset):
    def __init__(self,
                 data_path,
                 episode_length,
                 num_runs,
                 buffer_example,
                 img_size,
                 vel_command_dim=4,
                 normalized_img=False
                 ):
        self.data_path = data_path
        self.img_size = img_size
        # self.preprocess = T.Compose([T.ToTensor(),
        #                              T.Resize((self.img_size, self.img_size))])
        self.T = episode_length
        self.B = num_runs
        self.t = 0
        self.len = self.T * self.B
        self.vel_command_dim = vel_command_dim,
        self.samples = buffer_from_example(buffer_example, (self.T, self.B))
        self.offlinesamples = OfflineSamples
        self.current_line = None
        self.next_line = None
        self.normalized_img = normalized_img
        self.extract_data()

    # extract and tensorfy the data
    def extract_data(self):
        runs = glob.glob(str(self.data_path) + '/run*')
        runs.sort()
        # assert self.B == len(runs)
        for run_idx in tqdm(range(self.B)):
            run = runs[run_idx]
            actions = np.loadtxt(os.path.join(run, 'labels.csv'), delimiter=' ')
            img_names = glob.glob(os.path.join(self.data_path, run, 'images/*png'))
            img_names.sort()
            index = 0
            print(run)
            for i, img_name in enumerate(img_names):
                index += 1
                image = cv2.imread(img_name)  # note that the shape of image read by cv2 is (w, h, c)
                if self.normalized_img:
                    image = cv2.resize(image, (self.img_size, self.img_size))
                    image = T.ToTensor()(image)
                else:
                    image = cv2.resize(image, (self.img_size, self.img_size)).transpose((2, 0, 1))
                action = actions[i]
                self.current_line = actions[i]
                if index % self.T == 0:
                    self.next_line = actions[i]
                else:
                    self.next_line = actions[i + 1]

                trans, rotation = self.rotation_trans(self.current_line, self.next_line)
                velocity = action[-4:]  # extract the velocity from labels
                attitude_quad = action[:4]
                attitude_matrix = self.quad_to_matrix(attitude_quad)
                direction_label = self.extract_direction(velocity)
                samples = self.offlinesamples(observation=image,
                                              translation=np.array(trans, dtype=np.float32),
                                              rotation=np.array(rotation, dtype=np.float32),
                                              velocity=np.array(velocity, dtype=np.float32),
                                              direction=np.array(direction_label, dtype=np.float32),
                                              attitude=np.array(attitude_matrix, dtype=np.float32))
                self.samples[i][run_idx] = samples
                if index % self.T == 0:
                    # scale factor that divides all actions by max(x, y, z) across a run, for every run
                    scale_factors = np.max(np.abs(self.samples.translation[:, run_idx]), axis=0)
                    self.samples.translation[:, run_idx] /= scale_factors
                    print('had read num of images:', index)
                    break

    # for imitation learning
    def extract_img(self):
        runs = glob.glob(str(self.data_path) + '/run*')
        assert self.B == len(runs)
        runs.sort()
        for run_idx in tqdm(range(self.B)):
            run = runs[run_idx]
            actions = np.loadtxt(os.path.join(run, 'labels.txt'), delimiter=',')
            img_names = glob.glob(os.path.join(self.data_path, run, 'images/*png'))
            img_names.sort()
            index = 0
            for i, img_name in enumerate(img_names):
                index += 1
                image = cv2.imread(img_name)
                if self.normalized_img:
                    image = cv2.resize(image, (self.img_size, self.img_size))
                    image = T.ToTensor()(image)
                else:
                    image = cv2.resize(image, (self.img_size, self.img_size)).transpose((2, 0, 1))
                action = actions[i]
                velocity = action[:4]  # vx_body, vy_body, vz_body, v_yaw
                velocity = self.normalize_v(velocity)
                direction_label = self.extract_direction(velocity)
                attitude_quad = action[4:]  # qx, qy, qz, qw
                attitude_matrix = R.from_quat(attitude_quad).as_matrix()
                attitude_matrix = attitude_matrix.reshape(-1)
                # img_tensor = self.preprocess(image)
                sample = self.offlinesamples(observation=image,
                                             translation=np.zeros(3),
                                             rotation=np.zeros(6),
                                             velocity=np.array(velocity, dtype=np.float32),
                                             direction=np.array(direction_label, dtype=np.float32),
                                             attitude=np.array(attitude_matrix, dtype=np.float32)
                                             )
                self.samples[i][run_idx] = sample
                if index % self.T == 0:
                    print('had read num of images:', index)
                    break

    @staticmethod
    def normalize_v(v):
        # normalization of velocities from whatever to [-1, 1] range
        v_x_range = [-1, 7]
        v_y_range = [-3, 3]
        v_z_range = [-3, 3]
        v_yaw_range = [-1, 1]
        if len(v.shape) == 1:
            # means that it's a 1D vector of velocities
            v[0] = 2.0 * (v[0] - v_x_range[0]) / (v_x_range[1] - v_x_range[0]) - 1.0
            v[1] = 2.0 * (v[1] - v_y_range[0]) / (v_y_range[1] - v_y_range[0]) - 1.0
            v[2] = 2.0 * (v[2] - v_z_range[0]) / (v_z_range[1] - v_z_range[0]) - 1.0
            v[3] = 2.0 * (v[3] - v_yaw_range[0]) / (v_yaw_range[1] - v_yaw_range[0]) - 1.0
        elif len(v.shape) == 2:
            # means that it's a 2D vector of velocities
            v[:, 0] = 2.0 * (v[:, 0] - v_x_range[0]) / (v_x_range[1] - v_x_range[0]) - 1.0
            v[:, 1] = 2.0 * (v[:, 1] - v_y_range[0]) / (v_y_range[1] - v_y_range[0]) - 1.0
            v[:, 2] = 2.0 * (v[:, 2] - v_z_range[0]) / (v_z_range[1] - v_z_range[0]) - 1.0
            v[:, 3] = 2.0 * (v[:, 3] - v_yaw_range[0]) / (v_yaw_range[1] - v_yaw_range[0]) - 1.0
        else:
            raise Exception('Error in data format of V shape: {}'.format(v.shape))
        return v

    @staticmethod
    def de_normalize_v(v):
        # normalization of velocities from [-1, 1] range to whatever
        v_x_range = [-1, 7]
        v_y_range = [-3, 3]
        v_z_range = [-3, 3]
        v_yaw_range = [-1, 1]
        if len(v.shape) == 1:
            # means that it's a 1D vector of velocities
            v[0] = (v[0] + 1.0) / 2.0 * (v_x_range[1] - v_x_range[0]) + v_x_range[0]
            v[1] = (v[1] + 1.0) / 2.0 * (v_y_range[1] - v_y_range[0]) + v_y_range[0]
            v[2] = (v[2] + 1.0) / 2.0 * (v_z_range[1] - v_z_range[0]) + v_z_range[0]
            v[3] = (v[3] + 1.0) / 2.0 * (v_yaw_range[1] - v_yaw_range[0]) + v_yaw_range[0]
        elif len(v.shape) == 2:
            # means that it's a 2D vector of velocities
            v[:, 0] = (v[:, 0] + 1.0) / 2.0 * (v_x_range[1] - v_x_range[0]) + v_x_range[0]
            v[:, 1] = (v[:, 1] + 1.0) / 2.0 * (v_y_range[1] - v_y_range[0]) + v_y_range[0]
            v[:, 2] = (v[:, 2] + 1.0) / 2.0 * (v_z_range[1] - v_z_range[0]) + v_z_range[0]
            v[:, 3] = (v[:, 3] + 1.0) / 2.0 * (v_yaw_range[1] - v_yaw_range[0]) + v_yaw_range[0]
        else:
            raise Exception('Error in data format of V shape: {}'.format(v.shape))
        return v

    @staticmethod
    def rotation_trans(current_line, next_line):
        cQw, cQx, cQy, cQz, cPx, cPy, cPz = current_line[:7]
        nQw, nQx, nQy, nQz, nPx, nPy, nPz = next_line[:7]
        delta_p = np.array([cPx - nPx, cPy - nPy, cPz - nPz], dtype=np.float32)
        q1 = np.array([cQx, cQy, cQz, cQw], np.float32)
        q2 = np.array([nQx, nQy, nQz, nQw], np.float32)
        rot1 = R.from_quat(q1)
        rot2 = R.from_quat(q2)
        r = rot1.inv() * rot2
        r_matrix = r.as_matrix()
        real_angles_raw = r_matrix.tolist()
        r_matrix = np.delete(real_angles_raw, -1, 1)  # delete the last col of r_matrix
        six_dim_repre = r_matrix.reshape((1, -1))
        return delta_p, six_dim_repre

    def extract_direction(self, velocity):
        # we just need the vy, vz, vyaw for prediction
        vel_needed = velocity[1:]
        direction_sign = (np.sign(vel_needed) + 1.0) / 2.0
        label = 0
        for i, value in enumerate(direction_sign):
            label += np.power(2, i) * value
        return label

    def quad_to_matrix(self, attitude_quad):
        attitude_quad = np.roll(attitude_quad, -1)
        rot = R.from_quat(attitude_quad)
        r_matrix = rot.as_matrix()
        matrix_list = r_matrix.reshape(-1)
        return matrix_list

    def __len__(self):
        return self.len
