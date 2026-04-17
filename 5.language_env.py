
"""
5.language_env.py — Data collection orchestration for UR5e peg-in-hole
with language-aware features and force/torque sensing.

Usage:
    python 5.language_env.py [--num_demo 20] [--seed 0] [--root ./demo_data_pih]
"""

import sys
import os
import random
import argparse
import shutil
import numpy as np
from PIL import Image

from mujoco_env.pih_env2 import PIHEnv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description='PIH language-aware data collection')
    parser.add_argument('--num_demo', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--root', type=str, default='./demo_data_pih_ee')
    parser.add_argument('--repo_name', type=str, default='ur5e_pih_language')
    parser.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    return parser.parse_args()


def main():
    args = parse_args()

    SEED = args.seed
    NUM_DEMO = args.num_demo
    ROOT = args.root
    REPO_NAME = args.repo_name

    # Create environment
    PIHEnv = PIHEnv2(
        xml_path=args.xml_path,
        seed=SEED,
        state_type='joint_angle',
    )

    # 获取三个site的ID
    model = PIHEnv.env.model
    data = PIHEnv.env.data

    # 获取site的ID
    peg_tip_site_id = model.site("peg_tip_site").id
    hole_entry_site_id = model.site("hole_entry_site").id
    hole_bottom_site_id = model.site("hole_bottom_site").id

    print(f"Site IDs: peg_tip={peg_tip_site_id}, entry={hole_entry_site_id}, bottom={hole_bottom_site_id}")

    # Create or load dataset
    create_new = True
    if os.path.exists(ROOT):
        print(f"Directory {ROOT} already exists.")
        ans = input("Do you want to delete it? (y/n) ")
        if ans == 'y':
            shutil.rmtree(ROOT)
        else:
            create_new = False

    if create_new:
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            root=ROOT,
            robot_type="ur5e",
            fps=20,
            features={
                "observation.image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.wrist_image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["state"],
                },
                "observation.force_torque": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["force_torque"],  # fx, fy, fz, tx, ty, tz
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"],  # 6 joint angles + 1 gripper
                },
                "obj_init": {
                    "dtype": "float32",
                    "shape": (9,),
                    "names": ["obj_init"],  # peg(3) + hole_A(3) + hole_B(3)
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
    else:
        print("Load from previous dataset")
        dataset = LeRobotDataset(REPO_NAME, root=ROOT)

    # ------------------------------------------------------------------
    # Teleop loop
    # ------------------------------------------------------------------
    action = np.zeros(7)
    episode_id = 0
    record_flag = False

    while PIHEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
        PIHEnv.step_env()

        if PIHEnv.env.loop_every(HZ=20):
            # Check success
            done = PIHEnv.check_success()
            if done:
                dataset.save_episode()
                PIHEnv.reset()
                episode_id += 1
                record_flag = False
                print(f"Episode {episode_id}/{NUM_DEMO} saved.")

            # Teleop
            action, reset = PIHEnv.teleop_robot()

            if not record_flag and np.sum(np.abs(action[:6])) > 1e-6:
                record_flag = True
                print("Start recording")

            if reset:
                PIHEnv.reset()
                dataset.clear_episode_buffer()
                record_flag = False

            # Grab observations
            agent_image, wrist_image = PIHEnv.grab_image()
            agent_image = np.array(
                Image.fromarray(agent_image).resize((256, 256))
            )
            wrist_image = np.array(
                Image.fromarray(wrist_image).resize((256, 256))
            )

            # Step environment with action
            joint_q = PIHEnv.step(action)
            action = PIHEnv.q[:7]  # 6 joint angles and 1 gripper
            action = action.astype(np.float32)

            # # 获取site坐标并打印
            # peg_tip_pos = data.site_xpos[peg_tip_site_id]
            # hole_entry_pos = data.site_xpos[hole_entry_site_id]
            # hole_bottom_pos = data.site_xpos[hole_bottom_site_id]
            #
            # print(f"Peg tip site: {peg_tip_pos}")
            # print(f"Hole entry site: {hole_entry_pos}")
            # print(f"Hole bottom site: {hole_bottom_pos}")

            # Get EE pose and F/T
            ee_pose = PIHEnv.get_ee_pose()
            ft = PIHEnv.get_force_torque()

            if record_flag:
                dataset.add_frame(
                    {
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": joint_q[:6],
                        "observation.force_torque": ft,
                        "action": action,  # 6 joints + 1 gripper
                        "obj_init": PIHEnv.obj_init_pose,
                    },
                    task=PIHEnv.instruction,
                )

            PIHEnv.render(teleop=True, idx=episode_id)

    PIHEnv.env.close_viewer()

    # Clean up temp images if they exist
    images_dir = os.path.join(ROOT, 'images')
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)

    print(f"Data collection complete. {episode_id} episodes saved to {ROOT}")


if __name__ == '__main__':
    main()
