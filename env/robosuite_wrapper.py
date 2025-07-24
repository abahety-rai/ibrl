import os
import json
from robomimic.config import config_factory
import imageio

from typing import Optional, Any, Deque, Dict, Iterable
from collections import defaultdict, deque
from collections import OrderedDict

import torch
import robosuite
import matplotlib.pyplot as plt
# from robosuite import load_controller_config
import numpy as np
from common_utils import ibrl_utils as utils
from einops import rearrange
import common_utils

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

from tools.utils import is_image_corrupted


# all avail views:
# 'frontview', 'birdview', --> too far for this task
# 'agentview', 'robot0_robotview', --> same
# 'sideview', 'robot0_eye_in_hand'
GOOD_CAMERAS = {
    "Lift": ["agentview", "sideview", "robot0_eye_in_hand"],
    "PickPlaceCan": ["agentview", "robot0_eye_in_hand"],
    "NutAssemblySquare": ["agentview", "robot0_eye_in_hand"],
}
DEFAULT_CAMERA = "agentview"


DEFAULT_STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
STATE_KEYS = {
    "Lift": DEFAULT_STATE_KEYS,
    "PickPlaceCan": DEFAULT_STATE_KEYS,
    "NutAssemblySquare": DEFAULT_STATE_KEYS,
    "TwoArmTransport": [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot1_eef_pos",
        "robot1_eef_quat",
        "robot1_gripper_qpos",
        "object",
    ],
    "ToolHang": [
        "object",  # (389, 44)
        "robot0_eef_pos",  # (389, 3)
        "robot0_eef_quat",  # (389, 4)
        "robot0_gripper_qpos",  # (389, 2)
        # "robot0_gripper_qvel",  # (389, 2)
        # "robot0_eef_vel_ang",  # (389, 3)
        # "robot0_eef_vel_lin",  # (389, 3)
        # "robot0_joint_pos", # (389, 7)
        # "robot0_joint_pos_cos",  # (389, 7)
        # "robot0_joint_pos_sin",  # (389, 7)
        # "robot0_joint_vel",  # (389, 7)
    ],
    "PnPCounterToSink": [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos"
    ],
}
STATE_SHAPE = {
    "Lift": (19,),
    "PickPlaceCan": (23,),
    "NutAssemblySquare": (23,),
    "TwoArmTransport": (59,),
    "ToolHang": (53,),
}
PROP_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
PROP_DIM = 9


def create_key_to_history_mapping(
    obs_visual: Iterable[str], obs_state: Iterable[str], n_obs_history_visual: int, n_obs_history_state: int
) -> dict[str, int]:
    """Make dictionary mapping observation key to history length."""
    key_to_n_obs_history = dict()
    for k in obs_visual:
        key_to_n_obs_history[k] = n_obs_history_visual
    for k in obs_state:
        key_to_n_obs_history[k] = n_obs_history_state
    return key_to_n_obs_history

class PixelRobocasa:
    def __init__(self, env_name,
                     obs_stack=1,
                     use_state=False,
                     prop_stack=1,
                     cond_action=0,
                     device="cuda",
                     state_stack=1,
                     camera_names=[DEFAULT_CAMERA],
                     rl_cameras=["robot0_eye_in_hand"],
                     bc_cameras=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
                     image_size=512,
                     rl_image_size=96,
                     bc_image_size=128,
                     record_sim_state: bool = False,
                     max_episode_length: int = 100,
                     *args,
                     **kwargs):
        self.env_name = env_name
        
        # The task does not matter here, TODO: improve this
        robocasa_config = "/root/tmp/autogen_configs/ril/bc/robocasa/im/07-02-close_single_door_bc_xfmr_mg-3000/07-02-25-17-02-52/json/seed_123_ds_mg-3000.json"
        ext_cfg = json.load(open(robocasa_config, 'r'))
        robocasa_config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with robocasa_config.values_unlocked():
            robocasa_config.update(ext_cfg)
        dataset_cfg = robocasa_config.train.data[0]
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = robocasa_config.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
        deep_update(env_meta, robocasa_config.experiment.env_meta_update_dict)
        ObsUtils.initialize_obs_utils_with_config(robocasa_config)
        env_meta["env_kwargs"]["camera_heights"] = image_size
        env_meta["env_kwargs"]["camera_widths"] = image_size

        # TODO: remove this later
        layout, style = 0, 0
        env_meta["env_kwargs"]["layout_and_style_ids"] = [[layout, style]]

        shape_meta = {'ac_dim': 12, 'all_shapes': OrderedDict([('robot0_agentview_left_image', [3, 128, 128]), ('robot0_agentview_right_image', [3, 128, 128]), ('robot0_base_pos', [3]), ('robot0_base_quat', [4]), ('robot0_base_to_eef_pos', [3]), ('robot0_base_to_eef_quat', [4]), ('robot0_eye_in_hand_image', [3, 128, 128]), ('robot0_gripper_qpos', [2])]), 'all_obs_keys': ['robot0_agentview_left_image', 'robot0_agentview_right_image', 'robot0_base_pos', 'robot0_base_quat', 'robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos'], 'use_images': True}

        env_i = 0   
        env_kwargs = dict(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=robocasa_config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
                seed=robocasa_config.train.seed * 1000 + env_i,
            )
        self.env = EnvUtils.create_env_from_metadata(**env_kwargs)

        # Setting a fixed ep meta for now
        ep_meta = {'layout_id': 0, 'style_id': 0, 'object_cfgs': [{'name': 'obj', 'obj_groups': 'all', 'exclude_obj_groups': None, 'graspable': True, 'washable': True, 'placement': {'fixture': 'counter_main_main_group', 'sample_region_kwargs': {'ref': 'sink_main_group', 'loc': 'left_right'}, 'size': (0.3, 0.4), 'pos': ('ref', -1.0)}, 'type': 'object', 'info': {'groups_containing_sampled_obj': ['all', 'cucumber', 'vegetable', 'food', 'in_container', 'food_set1'], 'groups': ['all'], 'cat': 'cucumber', 'split': 'B', 'mjcf_path': '/mnt/proj-maple/abahety/robocasa/robocasa/models/assets/objects/objaverse/cucumber/cucumber_5/model.xml'}}, {'name': 'distr_counter', 'obj_groups': 'all', 'placement': {'fixture': 'counter_main_main_group', 'sample_region_kwargs': {'ref': 'sink_main_group', 'loc': 'left_right'}, 'size': (0.3, 0.3), 'pos': ('ref', -1.0), 'offset': (0.0, 0.3)}, 'type': 'object', 'info': {'groups_containing_sampled_obj': ['all', 'chips', 'packaged_food'], 'groups': ['all'], 'cat': 'chips', 'split': 'B', 'mjcf_path': '/mnt/proj-maple/abahety/robocasa/robocasa/models/assets/objects/objaverse/chips/chips_3/model.xml'}}, {'name': 'distr_sink', 'obj_groups': 'all', 'washable': True, 'placement': {'fixture': 'sink_main_group', 'size': (0.25, 0.25), 'pos': (0.0, 1.0)}, 'type': 'object', 'info': {'groups_containing_sampled_obj': ['all', 'mango', 'fruit', 'food', 'in_container'], 'groups': ['all'], 'cat': 'mango', 'split': 'B', 'mjcf_path': '/mnt/proj-maple/abahety/robocasa/robocasa/models/assets/objects/objaverse/mango/mango_4/model.xml'}}], 'fixtures': {'wall_room': {'cls': 'Wall'}, 'wall_backing_room': {'cls': 'Wall'}, 'wall_left_room': {'cls': 'Wall'}, 'wall_left_backing_room': {'cls': 'Wall'}, 'wall_right_room': {'cls': 'Wall'}, 'wall_right_backing_room': {'cls': 'Wall'}, 'floor_room': {'cls': 'Floor'}, 'floor_backing_room': {'cls': 'Floor'}, 'outlet_room': {'cls': 'WallAccessory'}, 'outlet_2_room': {'cls': 'WallAccessory'}, 'light_switch_room': {'cls': 'WallAccessory'}, 'light_switch_2_room': {'cls': 'WallAccessory'}, 'sink_main_group': {'cls': 'Sink'}, 'counter_main_main_group': {'cls': 'Counter'}, 'stove_main_group': {'cls': 'Stove'}, 'counter_right_main_group': {'cls': 'Counter'}, 'fridge_main_group': {'cls': 'Fridge'}, 'fridge_housing_main_group': {'cls': 'HousingCabinet'}, 'cab_1_main_group': {'cls': 'SingleCabinet'}, 'cab_2_main_group': {'cls': 'HingeCabinet'}, 'cab_main_main_group': {'cls': 'HingeCabinet'}, 'microwave_main_group': {'cls': 'Microwave'}, 'cab_micro_main_group': {'cls': 'HingeCabinet'}, 'cab_3_main_group': {'cls': 'OpenCabinet'}, 'cab_4_main_group': {'cls': 'HingeCabinet'}, 'stack_1_main_group_base': {'cls': 'Box'}, 'stack_1_main_group_1': {'cls': 'Drawer'}, 'stack_1_main_group_2': {'cls': 'Drawer'}, 'stack_1_main_group_3': {'cls': 'Drawer'}, 'stack_1_main_group_4': {'cls': 'Drawer'}, 'stack_2_main_group_base': {'cls': 'Box'}, 'stack_2_main_group_1': {'cls': 'HingeCabinet'}, 'stack_2_main_group_2': {'cls': 'PanelCabinet'}, 'dishwasher_main_group': {'cls': 'Dishwasher'}, 'stack_3_main_group_base': {'cls': 'Box'}, 'stack_3_main_group_1': {'cls': 'SingleCabinet'}, 'stack_4_main_group_base': {'cls': 'Box'}, 'stack_4_main_group_1': {'cls': 'SingleCabinet'}, 'stack_4_main_group_2': {'cls': 'Drawer'}, 'coffee_machine_main_group': {'cls': 'CoffeeMachine'}, 'toaster_main_group': {'cls': 'Toaster'}, 'knife_block_main_group': {'cls': 'Accessory'}, 'paper_towel_main_group': {'cls': 'Accessory'}}, 'gen_textures': {}, 'lang': 'pick the cucumber from the counter and place it in the sink', 'fixture_refs': {'sink': 'sink_main_group', 'counter': 'counter_main_main_group'}, 'cam_configs': {'robot0_agentview_center': {'pos': [-0.6, 0.0, 1.15], 'quat': [0.636945903301239, 0.3325185477733612, -0.3199238181114197, -0.6175596117973328], 'parent_body': 'mobilebase0_support'}, 'robot0_agentview_left': {'pos': [-0.5, 0.35, 1.05], 'quat': [0.55623853, 0.29935253, -0.37678665, -0.6775092], 'camera_attribs': {'fovy': '60'}, 'parent_body': 'mobilebase0_support'}, 'robot0_agentview_right': {'pos': [-0.5, -0.35, 1.05], 'quat': [0.6775091886520386, 0.3767866790294647, -0.2993525564670563, -0.55623859167099], 'camera_attribs': {'fovy': '60'}, 'parent_body': 'mobilebase0_support'}, 'robot0_frontview': {'pos': [-0.5, 0, 0.95], 'quat': [0.6088936924934387, 0.3814677894115448, -0.3673907518386841, -0.5905545353889465], 'camera_attribs': {'fovy': '60'}, 'parent_body': 'mobilebase0_support'}, 'robot0_eye_in_hand': {'pos': [0.05, 0, 0], 'quat': [0, 0.707107, 0.707107, 0], 'parent_body': 'robot0_right_hand'}}}
        self.env.env.set_ep_meta(ep_meta)

        self.use_state = use_state
        self.num_envs = 1
        self._observation_shape: tuple[int, ...] = (3 * obs_stack, rl_image_size, rl_image_size)
        self.prop_shape: tuple[int] = (PROP_DIM * prop_stack,)
        self.action_dim: int = len(self.env.env.action_spec[0])
        self.device = device

        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = True
        self.end_on_success = True
        self.env_reward_scale = 1.0
        self.max_episode_length = max_episode_length
        self.obs_stack = obs_stack
        self.state_stack = state_stack
        self.prop_stack = prop_stack
        self.cond_action = cond_action
        self.past_obses = defaultdict(list)
        self.past_actions = deque(maxlen=self.cond_action)

        self.state_keys = STATE_KEYS[env_name]
        self.prop_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        self.camera_names = camera_names
        self.bc_cameras = bc_cameras
        self.bc_image_keys = [f"{cam}_image" for cam in self.bc_cameras]
        self.rl_cameras = rl_cameras    
        self.rl_image_keys = [f"{cam}_image" for cam in self.rl_cameras]
        self.flip_image = False
        self.resize_transform = None
        self.bc_resize_transform = None
        self.image_size = image_size
        self.rl_image_size = rl_image_size or image_size
        self.bc_image_size = bc_image_size or image_size
        if self.rl_image_size != self.image_size:
            self.resize_transform = utils.get_rescale_transform(self.rl_image_size)
        
        if self.bc_image_size != self.image_size:
            self.bc_resize_transform = utils.get_rescale_transform(self.bc_image_size)
        
        self.record_sim_state = record_sim_state

        # visuomotor diffuision policy stuff
        self.obs_visual = ["color"]
        self.n_obs_history_visual = 2
        self.n_obs_history_state = 2
        self.key_to_n_obs_history = create_key_to_history_mapping(
            obs_visual=self.obs_visual,
            obs_state=self.state_keys,
            n_obs_history_visual=self.n_obs_history_visual,
            n_obs_history_state=self.n_obs_history_state,
        )

        self.video_writer = None
        self.video_count = 0
        self.video_skip = 5
        self.write_video = True
        self.frames = []
        self.video_path = "temp_videos/temp"
        os.makedirs(self.video_path, exist_ok=True)
        os.makedirs(f"{self.video_path}/temp_images", exist_ok=True)
        os.makedirs(f"{self.video_path}/temp_images2", exist_ok=True)


    @property
    def observation_shape(self):
        # if self.use_state:
        #     return self._state_shape
        # else:
        return self._observation_shape

    def reset(self, replay=None, unset_ep_meta=True, global_step=None) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
        self.past_obses.clear()
        self.past_actions.clear()
        for _ in range(self.cond_action):
            self.past_actions.append(torch.zeros(self.action_dim))

        all_obs = self.env.reset(unset_ep_meta=unset_ep_meta)
        rl_obs, high_res_images = self._extract_images(all_obs)

        if self.cond_action > 0:
            past_action = torch.from_numpy(np.stack(self.past_actions)).to(self.device)
            rl_obs["past_action"] = past_action

        if self.write_video:
            print("video writes to " + self.video_path)
            if replay is not None:
                self.video_writer = imageio.get_writer(f"{self.video_path}/demo_{replay.size()}.mp4", fps=20)
            else:
                if global_step is not None:
                    self.video_writer = imageio.get_writer(f"{self.video_path}/temp_{global_step}.mp4", fps=20)
                else:
                    self.video_writer = imageio.get_writer(f"{self.video_path}/temp.mp4", fps=20)


        return all_obs, rl_obs, high_res_images
    
    def extract_bc_obs(self, last_n_obs: Deque, task_description: list[str]) -> dict:
        bc_obs = self.process_n_obs(last_n_obs)
        bc_obs["task_description"] = task_description
        img = bc_obs["color"]

        # TODO: Make this dynamic. Prepare output tensor
        resized_imgs = torch.zeros(2, 3, 128, 128, 3).to(self.device)

        for t in range(img.shape[0]):  # history
            for c in range(img.shape[1]):  # camera
                # Convert to (H, W, C) -> (C, H, W)
                img_chw = img[t, c].permute(2, 0, 1)
                # Apply transform (expects PIL Image or tensor in (C, H, W))
                img_resized = self.bc_resize_transform(img_chw.to(self.device))
                # Convert back to (H, W, C)
                img_hwc = img_resized.permute(1, 2, 0)
                resized_imgs[t, c] = img_hwc

        # print(resized_imgs.shape)  # Should be (2, 3, 128, 128, 3)
        bc_obs["color"] = resized_imgs

        bc_obs = {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in bc_obs.items()
        }
        return bc_obs

    
    def _extract_images(self, obs):
        # assert self.frame_stack == 1, "frame stack not supported"

        high_res_images = {}
        rl_obs = {}

        if self.use_state:
            states = []
            for key in self.state_keys:
                if key == "object":
                    key = "object-state"
                states.append(obs[key])
            state = torch.from_numpy(np.concatenate(states).astype(np.float32))
            # first append, then concat
            self.past_obses["state"].append(state)
            rl_obs["state"] = utils.concat_obs(
                len(self.past_obses["state"]) - 1, self.past_obses["state"], self.state_stack
            ).to(self.device)

        props = []
        for key in self.prop_keys:
            props.append(obs[key])
        prop = torch.from_numpy(np.concatenate(props).astype(np.float32))
        # first append, then concat
        self.past_obses["prop"].append(prop)
        rl_obs["prop"] = utils.concat_obs(
            len(self.past_obses["prop"]) - 1, self.past_obses["prop"], self.prop_stack
        ).to(self.device)

        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            image_obs = obs[image_key]
            if self.flip_image:
                image_obs = image_obs[::-1]
            # breakpoint()
            # image_obs = torch.from_numpy(image_obs.copy()).permute([2, 0, 1])
            image_obs = torch.from_numpy(image_obs.copy())

            # keep the high-res version for rendering
            high_res_images[camera_name] = image_obs
            if camera_name not in self.rl_cameras:
                continue

            rl_image_obs = image_obs
            if self.resize_transform is not None:
                # set the device here because transform is 5x faster on GPU
                rl_image_obs = self.resize_transform(rl_image_obs.to(self.device))
            # first append, then concat
            self.past_obses[camera_name].append(rl_image_obs)
            rl_obs[camera_name] = utils.concat_obs(
                len(self.past_obses[camera_name]) - 1,
                self.past_obses[camera_name],
                self.obs_stack,
            )

        if self.record_sim_state:
            sim_state = self.env.sim.get_state().flatten()
            rl_obs["sim_state"] = torch.from_numpy(sim_state)
            for key in DEFAULT_STATE_KEYS:
                env_key = "object-state" if key == "object" else key
                rl_obs[key] = torch.from_numpy(obs[env_key])

        return rl_obs, high_res_images

    @property
    def batched(self) -> bool:
        return self.num_envs > 1
    
    def get_color(self, obs: dict):
        color = np.array([obs[image_key] for image_key in self.bc_image_keys])
        color = color.transpose(0, 2, 3, 1) # robocasa obs are (c, h, w), so we convert to (h, w, c)
        if self.batched:
            color = rearrange(color, "n b h w c -> b n h w c")
        return color
    
    def process_obs(self, obs: dict) -> dict:
        """
        Args:
            obs: dictionary of observations returned from each step of simulation.
        Return:
            Dictionary with input data for `predict_action` of a policy.
        """
        batch = {}

        batch["color"] = self.get_color(obs)

        for key in self.state_keys:
            batch[key] = obs[key]

        return batch
    
    def process_n_obs(self, last_n_obs: Deque) -> dict:
        """Args:
        last_n_obs: Iterable of last n observations
        """

        last_n_batches = defaultdict(list)
        for i_obs, obs in enumerate(last_n_obs):
            batch = self.process_obs(obs)
            for k, v in batch.items():
                n_obs_history = self.key_to_n_obs_history[k]
                if i_obs < n_obs_history:
                    arr = torch.from_numpy(v)
                    # Convert float64 to float32
                    # When unbatched, it is already float32.
                    # Ideally we'd figure out why unbatched state obs are float64
                    if arr.dtype == torch.float64:
                        arr = arr.to(torch.float32)
                    last_n_batches[k].append(arr)

        for k, v in last_n_batches.items():
            # If the observations already are batched, so that the leading dimension
            # is over the environments, insert our time dimension _after_ that
            # Otherwise, the time dimension should be the initial dimension
            if self.batched:
                last_n_batches[k] = torch.stack(v, dim=1)
            else:
                last_n_batches[k] = torch.stack(v, dim=0)

        return last_n_batches

    def step(self, actions: torch.Tensor, replay=None, last_n_obs=None, curr_rl_obs=None, curr_bc_obs=None) -> tuple[dict, float, bool, bool, dict]:
        """
        all inputs and outputs are tensors
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        num_action = actions.size(0)

        rl_obs, bc_obs = {}, {}
        # record the action in original format from model
        if self.cond_action > 0:
            for i in range(actions.size(0)):
                self.past_actions.append(actions[i])
            past_action = torch.stack(list(self.past_actions)).to(self.device)
            rl_obs["past_action"] = past_action

        actions = actions.numpy()

        reward = 0
        success = False
        terminal = False
        high_res_images = {}
        for i in range(num_action):
            self.time_step += 1
            all_obs, step_reward, terminal, _ = self.env.step(actions[i])

            
            # if is_image_corrupted(all_obs["robot0_agentview_left_image"], debug=True):
            #     print("Likely corrupted or noisy!!!!!")
            #     img = all_obs["robot0_agentview_left_image"]
            #     img = np.transpose(img, (1, 2, 0))
            #     # print(f"--- Image min: {img.min()}, max: {img.max()}, dtype: {img.dtype}")
            #     plt.imsave(f"{self.video_path}/temp_images/temp_{self.time_step}.jpg", img)

            #     terminal = True
            #     reward = 0.0
            #     success = False
            #     return all_obs, reward, terminal, success, high_res_images

            # # remove later
            # if self.time_step % self.video_skip == 0:
            #     img = all_obs["robot0_agentview_left_image"]
            #     img = np.transpose(img, (1, 2, 0))
            #     # print(f"--- Image min: {img.min()}, max: {img.max()}, dtype: {img.dtype}")
            #     img_uint8 = (img * 255).round().astype(np.uint8)
            #     plt.imsave(f"{self.video_path}/temp_images/temp_{self.time_step}.png", img_uint8)
            
            # NOTE: extract images every step for potential obs stacking
            # this is not efficient
            # curr_rl_obs, curr_high_res_images = self._extract_images(all_obs)

            # if i == num_action - 1:
            #     rl_obs.update(curr_rl_obs)
            #     high_res_images.update(curr_high_res_images)

            # visualization
            if self.video_writer is not None:
                if self.time_step % self.video_skip == 0:
                    # frame = env.render(mode="rgb_array", height=512, width=512)
                    cam_imgs = []
                    for im_name in ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"]:
                        im = TensorUtils.to_numpy(all_obs[im_name])
                        im = np.transpose(im, (1, 2, 0))
                        cam_imgs.append(im)
                    frame = np.concatenate(cam_imgs, axis=1)
                    frame = (frame * 255.0).astype(np.uint8)
                    self.frames.append(frame)

            reward += step_reward
            self.episode_reward += step_reward

            if step_reward == 1:
                success = True
                if self.end_on_success:
                    terminal = True

            # print("time_step: ", self.time_step,  self.max_episode_length)
            if self.time_step == self.max_episode_length:
                terminal = True

            # NOTE: since action horizon can be >1, we add to the replay buffer here
            if replay is not None:
                reply = {"action": torch.tensor(actions[i])}
                replay.add(curr_rl_obs, curr_bc_obs, reply, reward, terminal, success, image_obs=None)
            # print("replay episode len: ", replay.episode.len())

            if last_n_obs is not None:
                self.shift_observation_window_by_one(last_n_obs, all_obs)

            if terminal:
                if self.video_writer is not None:
                    # imageio.mimsave(f"{video_path}/temp.mp4", self.frames, fps=20)
                    # self.video_writer = None
                    if self.video_writer is not None:
                        for frame in self.frames:
                            self.video_writer.append_data(frame)
                        self.video_writer.close()
                    # breakpoint()
                    self.frames = []
                break

        reward = reward * self.env_reward_scale
        self.terminal = terminal
        return all_obs, reward, terminal, success, high_res_images

    def shift_observation_window_by_one(self, last_n_obs: Deque, new_obs: dict) -> None:
        last_n_obs.appendleft(new_obs)
        last_n_obs.pop()


class PixelRobosuite:
    def __init__(
        self,
        env_name,
        robots,
        episode_length,
        *,
        reward_shaping=False,
        image_size=224,
        rl_image_size=96,
        device="cuda",
        camera_names=[DEFAULT_CAMERA],
        rl_cameras=["agentview"],
        env_reward_scale=1.0,
        end_on_success=True,
        use_state=False,
        obs_stack=1,
        state_stack=1,
        prop_stack=1,
        cond_action=0,
        flip_image=True,  # only false if using with eval_with_init_state
        ctrl_delta=True,
        record_sim_state: bool = False,
    ):
        assert isinstance(camera_names, list)
        self.camera_names = camera_names
        self.ctrl_config = load_controller_config(default_controller="OSC_POSE")
        self.ctrl_config["control_delta"] = ctrl_delta
        self.record_sim_state = record_sim_state
        self.env = robosuite.make(
            env_name=env_name,
            robots=robots,
            controller_configs=self.ctrl_config,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            reward_shaping=reward_shaping,
            camera_names=self.camera_names,
            camera_heights=image_size,
            camera_widths=image_size,
            horizon=episode_length,
        )
        self.rl_cameras = rl_cameras if isinstance(rl_cameras, list) else [rl_cameras]
        self.image_size = image_size
        self.rl_image_size = rl_image_size or image_size
        self.env_reward_scale = env_reward_scale
        self.end_on_success = end_on_success
        self.use_state = use_state
        self.state_keys = STATE_KEYS[env_name]
        self.prop_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        self.flip_image = flip_image

        self.resize_transform = None
        if self.rl_image_size != self.image_size:
            self.resize_transform = utils.get_rescale_transform(self.rl_image_size)

        self.action_dim: int = len(self.env.action_spec[0])
        self._observation_shape: tuple[int, ...] = (3 * obs_stack, rl_image_size, rl_image_size)
        self._state_shape: tuple[int] = (STATE_SHAPE[env_name][0] * state_stack,)
        self.prop_shape: tuple[int] = (PROP_DIM * prop_stack,)
        self.device = device

        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = True

        self.obs_stack = obs_stack
        self.state_stack = state_stack
        self.prop_stack = prop_stack
        self.cond_action = cond_action
        self.past_obses = defaultdict(list)
        self.past_actions = deque(maxlen=self.cond_action)

    @property
    def observation_shape(self):
        if self.use_state:
            return self._state_shape
        else:
            return self._observation_shape

    def _extract_images(self, obs):
        # assert self.frame_stack == 1, "frame stack not supported"

        high_res_images = {}
        rl_obs = {}

        if self.use_state:
            states = []
            for key in self.state_keys:
                if key == "object":
                    key = "object-state"
                states.append(obs[key])
            state = torch.from_numpy(np.concatenate(states).astype(np.float32))
            # first append, then concat
            self.past_obses["state"].append(state)
            rl_obs["state"] = utils.concat_obs(
                len(self.past_obses["state"]) - 1, self.past_obses["state"], self.state_stack
            ).to(self.device)

        props = []
        for key in self.prop_keys:
            props.append(obs[key])
        prop = torch.from_numpy(np.concatenate(props).astype(np.float32))
        # first append, then concat
        self.past_obses["prop"].append(prop)
        rl_obs["prop"] = utils.concat_obs(
            len(self.past_obses["prop"]) - 1, self.past_obses["prop"], self.prop_stack
        ).to(self.device)

        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            image_obs = obs[image_key]
            if self.flip_image:
                image_obs = image_obs[::-1]
            image_obs = torch.from_numpy(image_obs.copy()).permute([2, 0, 1])

            # keep the high-res version for rendering
            high_res_images[camera_name] = image_obs
            if camera_name not in self.rl_cameras:
                continue

            rl_image_obs = image_obs
            if self.resize_transform is not None:
                # set the device here because transform is 5x faster on GPU
                rl_image_obs = self.resize_transform(rl_image_obs.to(self.device))
            # first append, then concat
            self.past_obses[camera_name].append(rl_image_obs)
            rl_obs[camera_name] = utils.concat_obs(
                len(self.past_obses[camera_name]) - 1,
                self.past_obses[camera_name],
                self.obs_stack,
            )

        if self.record_sim_state:
            sim_state = self.env.sim.get_state().flatten()
            rl_obs["sim_state"] = torch.from_numpy(sim_state)
            for key in DEFAULT_STATE_KEYS:
                env_key = "object-state" if key == "object" else key
                rl_obs[key] = torch.from_numpy(obs[env_key])

        return rl_obs, high_res_images

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.time_step = 0
        self.episode_reward = 0
        self.episode_extra_reward = 0
        self.terminal = False
        self.past_obses.clear()
        self.past_actions.clear()
        for _ in range(self.cond_action):
            self.past_actions.append(torch.zeros(self.action_dim))

        obs = self.env.reset()
        rl_obs, high_res_images = self._extract_images(obs)

        if self.cond_action > 0:
            past_action = torch.from_numpy(np.stack(self.past_actions)).to(self.device)
            rl_obs["past_action"] = past_action

        return rl_obs, high_res_images

    def step(self, actions: torch.Tensor) -> tuple[dict, float, bool, bool, dict]:
        """
        all inputs and outputs are tensors
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        num_action = actions.size(0)

        rl_obs = {}
        # record the action in original format from model
        if self.cond_action > 0:
            for i in range(actions.size(0)):
                self.past_actions.append(actions[i])
            past_action = torch.stack(list(self.past_actions)).to(self.device)
            rl_obs["past_action"] = past_action

        actions = actions.numpy()

        reward = 0
        success = False
        terminal = False
        high_res_images = {}
        for i in range(num_action):
            self.time_step += 1
            obs, step_reward, terminal, _ = self.env.step(actions[i])
            # NOTE: extract images every step for potential obs stacking
            # this is not efficient
            curr_rl_obs, curr_high_res_images = self._extract_images(obs)

            if i == num_action - 1:
                rl_obs.update(curr_rl_obs)
                high_res_images.update(curr_high_res_images)

            reward += step_reward
            self.episode_reward += step_reward

            if step_reward == 1:
                success = True
                if self.end_on_success:
                    terminal = True

            if terminal:
                break

        reward = reward * self.env_reward_scale
        self.terminal = terminal
        return rl_obs, reward, terminal, success, high_res_images


if __name__ == "__main__":
    from torchvision.utils import save_image

    env = PixelRobosuite("Lift", "Panda", 200, image_size=256, camera_names=GOOD_CAMERAS["Lift"])
    x = env.reset()[0][GOOD_CAMERAS["Lift"][0]].float() / 255
    print(x.dtype)
    print(x.shape)
    save_image(x, "test_env.png")
