from env.robosuite_wrapper import PixelRobocasa
from tools.utils import is_image_corrupted
import numpy as np
import torch
import imageio
import os

import matplotlib.pyplot as plt

task_name = "PnPCounterToSink"
camera_names = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
rl_cameras = ["robot0_eye_in_hand"]
bc_cameras = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
episode_length = 300

train_env = PixelRobocasa(
    env_name=task_name,
    camera_names=camera_names,
    rl_cameras=rl_cameras,
    bc_cameras=bc_cameras,
    max_episode_length=episode_length,
)

train_env.reset()
video_path = f"temp_videos/investigate_rendering"
os.makedirs(video_path, exist_ok=True)

ep_corrupted = []
for ep_num in range(10):
    video_writer = imageio.get_writer(f"{video_path}/episode_{ep_num}.mp4", fps=20)
    for i in range(episode_length):
        action = np.random.uniform(-1, 1, size=(1, 12))
        action = torch.from_numpy(action).float()
        # action = train_env.action_space.sample()
        all_obs, reward, terminal, success, image_obs = train_env.step(action)
        img = all_obs["robot0_agentview_left_image"]
        img = np.transpose(img, (1, 2, 0))
        img = (img * 255.0).astype(np.uint8)
        video_writer.append_data(img)
        is_corrupted = is_image_corrupted(all_obs["robot0_agentview_left_image"], debug=True)
        if is_corrupted:
            break
            
    if is_corrupted:
        ep_corrupted.append(ep_num)
        is_corrupted = False
        #     plt.imsave(f"{video_path}/temp_images/temp_{i}.jpg", img)

    video_writer.close()

print(ep_corrupted)