import torch
from rlpyt.ul.models.ul.encoders import DmlabEncoderModelNorm
from rlpyt.models.mlp import MlpModel
from torchvision import transforms as T
from scipy.spatial.transform import Rotation


class VelRegressor:
    def __init__(self,
                 bc_weights_path,
                 image_shape=(3, 84, 84),
                 latent_size=256,
                 hidden_size=512,  # 2048 for byol/random encoder 512 for atc
                 mlp_hidden_layers=(128, 64, 16),
                 action_diim=4,
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
        self.policy = MlpModel(
            input_size=self.encoder.output_size+attitude_dim,
            hidden_sizes=list(mlp_hidden_layers),
            output_size=action_diim
        )
        loaded_state_dict = torch.load(self.bc_weight_path, map_location=torch.device('cpu'))
        loaded_state_dict = loaded_state_dict.get('algo_state_dict', loaded_state_dict)
        encoder_state_dict = loaded_state_dict.get('encoder', loaded_state_dict)
        policy_dict = loaded_state_dict.get('mlp_head', loaded_state_dict)
        self.encoder.load_state_dict(encoder_state_dict)
        self.policy.load_state_dict(policy_dict)

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
        bc_weights_path=f'/home/yibo/Documents/rlpyt/data/local/20220327/181107/mst_vel_regressor/learning_rate_0.0005/itr_19999.pkl',
        attitude_dim=0
    )
    atti_added_policy = VelRegressor(
        bc_weights_path=f'/home/yibo/Documents/rlpyt/data/local/20220401/231139/mst_vel_regressor/learning_rate_0.0005/itr_31999.pkl',
        attitude_dim=9
    )
