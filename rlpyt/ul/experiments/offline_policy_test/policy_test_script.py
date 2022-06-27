import torch
import numpy as np
import matplotlib.pyplot as plt
from rlpyt.ul.models.ul.encoders import DmlabEncoderModelNorm
from rlpyt.models.mlp import MlpModel
from torchvision import transforms as T
from scipy.spatial.transform import Rotation
from rlpyt.ul.replays.offline_ul_replay import OfflineUlReplayBuffer
from rlpyt.ul.replays.offline_dataset import OfflineDatasets

class VelRegressor:
    def __init__(self,
                 bc_weights_path,
                 image_shape=(3, 84, 84),
                 latent_size=256,
                 hidden_size=512,  # 2048 for byol/random encoder 512 for atc
                 mlp_hidden_layers=(128, 64, 16),
                 state_latent_dim=256,
                 action_dim=4,
                 attitude_dim=9,
                 ):
        self.image_shape = image_shape
        self.device = torch.device('cuda')
        self.bc_weight_path = bc_weights_path
        self.preprocess = T.ToTensor()
        self.encoder = DmlabEncoderModelNorm(
            image_shape=image_shape,
            latent_size=latent_size,
            hidden_sizes=hidden_size,
        )
        self.state_projector = torch.nn.Linear(
            in_features=attitude_dim,
            out_features=state_latent_dim,
            bias=False
        )
        self.policy = MlpModel(
            input_size=self.encoder.output_size+state_latent_dim,
            hidden_sizes=list(mlp_hidden_layers),
            output_size=action_dim
        )
        loaded_state_dict = torch.load(self.bc_weight_path, map_location=torch.device('cpu'))
        loaded_state_dict = loaded_state_dict.get('algo_state_dict', loaded_state_dict)
        encoder_state_dict = loaded_state_dict.get('encoder', loaded_state_dict)
        policy_dict = loaded_state_dict.get('mlp_head', loaded_state_dict)
        state_projector_dict = loaded_state_dict.get('state_projector', loaded_state_dict)
        self.encoder.load_state_dict(encoder_state_dict)
        self.policy.load_state_dict(policy_dict)
        self.state_projector.load_state_dict(state_projector_dict)

    @staticmethod
    def calculate_v_stats(predictions, v_gt):  # predictions (num_samples, predictions)
        # display averages
        mean_pred = np.mean(predictions, axis=0)
        mean_v = np.mean(v_gt, axis=0)
        print('Means (prediction, GT) : Vx({} , {}) Vy({} , {}) Vz({} , {}) Vyaw({} , {})'.format(
            mean_pred[0], mean_v[0], mean_pred[1], mean_v[1], mean_pred[2], mean_v[2], mean_pred[3], mean_v[3]))
        # display mean absolute error
        abs_diff = np.abs(predictions - v_gt)
        mae = np.mean(abs_diff, axis=0)
        print('Absolute errors : Vx({}) Vy({}) Vz({}) Vyaw({})'.format(mae[0], mae[1], mae[2], mae[3]))
        # display max errors
        max_diff = np.max(abs_diff, axis=0)
        print('Max error : Vx({}) Vy({}) Vz({}) Vyaw({})'.format(max_diff[0], max_diff[1], max_diff[2], max_diff[3]))
        plt.title("Vx Absolute Error histogram")
        _ = plt.hist(abs_diff[:, 0], np.linspace(0.0, 10.0, num=1000))
        plt.show()
        plt.title("Vy Absolute Error histogram")
        _ = plt.hist(abs_diff[:, 1], np.linspace(0.0, 3, num=1000))
        plt.show()
        plt.title("Vz Absolute Error histogram")
        _ = plt.hist(abs_diff[:, 2], np.linspace(0.0, 3, num=1000))
        plt.show()
        plt.title("Vyaw Absolute Error histogram")
        _ = plt.hist(abs_diff[:, 3], np.linspace(0.0, 3, num=1000))
        plt.show()

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

    def eval(self):
        self.encoder.eval()
        self.state_projector.eval()
        self.policy.eval()

    # def predict_velocities(self, img, labels, with_attitude=True):
    #     if with_attitude:
    #         quad = labels[]
    #         drone_attitude = Rotation.from_quat([quad.x_val, quad.y_val, quad.z_val, quad.w_val])
    #         drone_attitude = drone_attitude.as_matrix().reshape(-1).astype(np.float32)
    #     self.encoder.eval()
    #     self.policy.eval()
    #     with torch.no_grad():
    #         h = self.encoder.conv(img).reshape(1, -1)
    #         if state_estimated is not None:
    #             policy_input = torch.cat((h, torch.from_numpy(drone_attitude).unsqueeze(0)), dim=-1)
    #         else:
    #             policy_input = h
    #         predictions = self.policy(policy_input)
    #     predictions = predictions.detach().numpy()
    #     predictions = racing_utils.dataset_utils.de_normalize_v(predictions)
    #     # print('Predicted body vel: \n {}'.format(predictions[0]))
    #     v_xyz_world = racing_utils.geom_utils.convert_t_body_2_world(airsimdroneracingvae.Vector3r(predictions[0, 0],
    #                                                                                                predictions[0, 1],
    #                                                                                                predictions[0, 2]),
    #                                                                  p_o_b.orientation)
    #     return np.array([v_xyz_world.x_val, v_xyz_world.y_val, v_xyz_world.z_val, predictions[0, 3]])


if __name__ == '__main__':
    img_policy = VelRegressor(
        bc_weights_path=f'/home/yibo/Documents/rlpyt/data/local/20220621/154548/mst_state_mlp_vel_regressor/state_latent_dim_256_nostack/itr_29999.pkl',
    )
    img_policy.eval()
    dataloader = OfflineUlReplayBuffer(
        replay_buffer=OfflineDatasets,
        img_size=84,
        frame_stacks=1,
        data_path=f'/home/yibo/spaces/datasets/il_val_datasets',
        episode_length=4096,
        num_runs=1,
        forward_step=0,
    )
    samples = dataloader.sample_batch(batch_size=4096)
    obs = samples.observations
    length, batch_size, f, c, h, w = obs.shape
    if obs.dtype == torch.uint8:
        default_float_type = torch.get_default_dtype()
        obs = obs.to(dtype=default_float_type).div(255.0)

    vel_gt = samples.velocities
    attitude_states = samples.attitudes

    with torch.no_grad():
        conv_out = img_policy.encoder.conv(obs.reshape(length * batch_size * f, c, h, w))
        state_embedding = img_policy.state_projector(attitude_states)
        policy_input = torch.cat((conv_out.reshape(length * batch_size * f, -1),
                                  state_embedding.reshape(length * batch_size, -1)), dim=-1)
        pred_vels = img_policy.policy(policy_input)
    pred_vels = pred_vels.detach().numpy()
    denormalized_pred_vels = img_policy.de_normalize_v(pred_vels)
    denormalized_gt_vels = img_policy.de_normalize_v(vel_gt.squeeze().numpy())
    img_policy.calculate_v_stats(denormalized_pred_vels, denormalized_gt_vels)

