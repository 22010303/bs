
"""
7.staged_collect.py — Staged data collection for UR5e peg-in-hole.

Two-phase control:
  Phase 1 (auto-align): Controller automatically moves EEF above the target hole.
  Phase 2 (teleop insert): Keyboard control for peg insertion.

Hole positions are randomized (x, y) at each episode reset within the
robot's reachable workspace. The two holes stay close but do not overlap.

Usage:
    python 7.staged_collect.py [--num_demo 20] [--seed 0] [--root ./demo_data_staged]
"""

import sys
import os
import random
import argparse
import shutil
import numpy as np
from PIL import Image

from mujoco_env.pih_env2 import PIHEnv2
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# # --------------- Randomisation constants ---------------
# # Robot base is at x=-0.4, y=0.  Holes default at x=0.1.
# # Reachable workspace for hole placement (on table surface):
# HOLE_X_RANGE = (-0.05, 0.25)   # forward / backward on table
# HOLE_Y_RANGE = (-0.25, 0.25)   # left / right
# HOLE_MIN_SEP = 0.15            # minimum centre-to-centre distance between holes
# HOLE_MAX_SEP = 0.35            # keep them reasonably close
HOLE_Z = 1.065                 # fixed z (sitting on table)
#
# Height above hole for auto-align target
ALIGN_HEIGHT_ABOVE_HOLE = 0.35  # EEF will be this far above hole_z


def parse_args():
    parser = argparse.ArgumentParser(description='Staged PIH data collection')
    parser.add_argument('--num_demo', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--root', type=str, default='./demo_data_staged')
    parser.add_argument('--repo_name', type=str, default='ur5e_pih_staged')
    parser.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    return parser.parse_args()


# def random_hole_positions(rng):
#     """Sample two hole (x, y) positions that are close but not overlapping."""
#     for _ in range(1000):
#         x_A = rng.uniform(*HOLE_X_RANGE)
#         y_A = rng.uniform(*HOLE_Y_RANGE)
#         x_B = rng.uniform(*HOLE_X_RANGE)
#         y_B = rng.uniform(*HOLE_Y_RANGE)
#         dist = np.hypot(x_A - x_B, y_A - y_B)
#         if HOLE_MIN_SEP <= dist <= HOLE_MAX_SEP:
#             return (x_A, y_A), (x_B, y_B)
#     # Fallback: deterministic pair
#     return (0.1, -0.1), (0.1, 0.1)
def random_hole_positions(rng):
    """直接在指定范围内随机采样两个孔的位置"""
    # 孔A的固定范围
    x_A = rng.uniform(0, 0.10)  # 您想要的孔A的X范围
    y_A = rng.uniform(0.1, 0.2)  # 孔A的Y范围

    # 孔B的固定范围
    x_B = rng.uniform(0, 0.10)  # 孔B的X范围
    y_B = rng.uniform(-0.20, -0.10)  # 孔B的Y范围

    return (x_A, y_A), (x_B, y_B)


def main():
    args = parse_args()

    SEED = args.seed
    NUM_DEMO = args.num_demo
    ROOT = args.root
    REPO_NAME = args.repo_name

    rng = np.random.default_rng(SEED)

    # Create environment
    PIHEnv = PIHEnv2(
        xml_path=args.xml_path,
        seed=SEED,
        state_type='joint_angle',
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
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
                    "names": ["force_torque"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"],
                },
                "obj_init": {
                    "dtype": "float32",
                    "shape": (9,),
                    "names": ["obj_init"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
    else:
        print("Load from previous dataset")
        dataset = LeRobotDataset(REPO_NAME, root=ROOT)

    # ------------------------------------------------------------------
    # Helper: randomise hole positions and reset
    # ------------------------------------------------------------------
    # F/T zero-offset (calibrated at start of each episode)
    ft_offset = np.zeros(6, dtype=np.float32)

    def get_ft_calibrated():
        """Return force/torque with zero-offset subtracted."""
        return PIHEnv.get_force_torque() - ft_offset

    def calibrate_ft(n_samples=50):
        """Hold still and average F/T readings to compute zero offset."""
        nonlocal ft_offset
        readings = []
        for _ in range(n_samples):
            PIHEnv.step_env()
            readings.append(PIHEnv.get_force_torque())
        ft_offset = np.mean(readings, axis=0).astype(np.float32)
        print(f"  F/T zero-offset: F=[{ft_offset[0]:.2f},{ft_offset[1]:.2f},{ft_offset[2]:.2f}] "
              f"T=[{ft_offset[3]:.3f},{ft_offset[4]:.3f},{ft_offset[5]:.3f}]")

    def randomise_and_reset():
        """Randomise hole positions, reset env, calibrate F/T, return target hole position."""
        (xA, yA), (xB, yB) = random_hole_positions(rng)

        # Move hole bodies in the MuJoCo model
        PIHEnv.env.set_p_body('hole_A', np.array([xA, yA, HOLE_Z]), forward=False)
        PIHEnv.env.set_p_body('hole_B', np.array([xB, yB, HOLE_Z]), forward=False)

        # Reset arm to home + choose random peg
        PIHEnv.reset()

        # Calibrate F/T sensor zero offset
        calibrate_ft()

        # Return target hole position for the active peg
        target_name = PIHEnv.obj_target  # 'hole_A' or 'hole_B'
        p_hole = PIHEnv.env.get_p_body(target_name)
        print(f"  hole_A -> ({xA:.3f}, {yA:.3f})")
        print(f"  hole_B -> ({xB:.3f}, {yB:.3f})")
        print(f"  Active peg: {PIHEnv.active_peg}, target: {target_name}")
        return p_hole

    # ------------------------------------------------------------------
    # Auto-align state (Phase 1 runs step-by-step inside the 20Hz loop)
    # ------------------------------------------------------------------
    ALIGN_INTERP_STEPS = 100   # interpolation steps
    ALIGN_SETTLE_STEPS = 20    # settle steps after reaching target
    align_state = {}           # mutable dict to hold alignment progress

    def init_align(p_hole):
        """Compute IK target and prepare interpolation state."""
        target_pos = np.array([
            p_hole[0] + 0.005,
            p_hole[1] - 0.002,
            HOLE_Z + ALIGN_HEIGHT_ABOVE_HOLE,
        ])
        target_rot = rpy2r(np.deg2rad([180, 0, 0]))

        q_current = PIHEnv.env.get_qpos_joints(joint_names=PIHEnv.joint_names)
        q_target, _, _ = solve_ik(
            env=PIHEnv.env,
            joint_names_for_ik=PIHEnv.joint_names,
            body_name_trgt=PIHEnv.EEF_BODY,
            q_init=q_current,
            p_trgt=target_pos,
            R_trgt=target_rot,
            max_ik_tick=200,
            ik_stepsize=1.0,
            ik_eps=1e-2,
            ik_th=np.radians(5.0),
        )

        align_state['q_start'] = q_current.copy()
        align_state['q_target'] = q_target.copy()
        align_state['target_pos'] = target_pos
        align_state['target_rot'] = target_rot
        align_state['step'] = 0
        align_state['total'] = ALIGN_INTERP_STEPS + ALIGN_SETTLE_STEPS

    def step_align():
        """Advance one alignment step. Returns True when finished."""
        s = align_state['step']
        align_state['step'] += 1

        if s < ALIGN_INTERP_STEPS:
            # Interpolation phase
            alpha = (s + 1) / ALIGN_INTERP_STEPS
            q_interp = align_state['q_start'] * (1 - alpha) + align_state['q_target'] * alpha
        else:
            # Settle phase — hold at target
            q_interp = align_state['q_target']

        PIHEnv.q = np.concatenate([q_interp, [200.0]])
        PIHEnv.step_env()

        done = (align_state['step'] >= align_state['total'])
        if done:
            # Update internal state so teleop continues from here
            PIHEnv.p0 = align_state['target_pos'].copy()
            PIHEnv.R0 = align_state['target_rot'].copy()
            PIHEnv.last_q = align_state['q_target'].copy()
            PIHEnv.q = np.concatenate([align_state['q_target'], [200.0]])
            print(">>> 已到达hole上方 <<<")
            print("    现在可以使用键盘控制进行插入操作")
        return done

    def print_ft_metrics():
        """Print calibrated force/torque metrics."""
        ft = get_ft_calibrated()
        lateral_force = np.sqrt(ft[0] ** 2 + ft[1] ** 2)
        axial_force = abs(ft[2])
        total_force = np.linalg.norm(ft[:3])
        bending_torque = np.sqrt(ft[3] ** 2 + ft[4] ** 2)
        twist_torque = abs(ft[5])
        total_torque = np.linalg.norm(ft[3:])
        print(f"  [校准F/T] lateral={lateral_force:.2f} axial={axial_force:.2f} total_F={total_force:.2f} | "
              f"bend={bending_torque:.3f} twist={twist_torque:.3f} total_T={total_torque:.3f}")

    def check_success_calibrated():
        """check_success using calibrated F/T instead of raw values."""
        peg_body = PIHEnv.PEG_CONFIG[PIHEnv.active_peg]['body']
        p_peg = PIHEnv.env.get_p_body(peg_body)
        p_hole = PIHEnv.env.get_p_body(PIHEnv.obj_target)
        hole_top_z = PIHEnv.HOLE_OPENING_Z[PIHEnv.obj_target]

        xy_dist = np.linalg.norm(p_peg[:2] - p_hole[:2])
        peg_below_opening = p_peg[2] < (hole_top_z - 0.02)

        if not (xy_dist < 0.03 and peg_below_opening):
            return False

        ft = get_ft_calibrated()
        lateral_force = np.sqrt(ft[0] ** 2 + ft[1] ** 2)
        axial_force = abs(ft[2])
        total_force = np.linalg.norm(ft[:3])
        bending_torque = np.sqrt(ft[3] ** 2 + ft[4] ** 2)
        twist_torque = abs(ft[5])
        total_torque = np.linalg.norm(ft[3:])

        thresholds = {
            'lateral_force': 20.0,
            'axial_force_max': 80.0,
            'total_force_max': 80.0,
            'bending_torque': 1.5,
            'twist_torque': 1.5,
            'total_torque': 2.0,
        }

        if (lateral_force > thresholds['lateral_force'] or
                axial_force > thresholds['axial_force_max'] or
                total_force > thresholds['total_force_max'] or
                bending_torque > thresholds['bending_torque'] or
                twist_torque > thresholds['twist_torque'] or
                total_torque > thresholds['total_torque']):
            print_ft_metrics()
            print("  => 力过大，判定失败")
            return False

        if axial_force < 1.0:
            return False

        print_ft_metrics()
        print("  => SUCCESS!")
        return True

    def render_with_calibrated_ft(teleop=False, idx=0):
        """Render with calibrated F/T overlay instead of raw values."""
        PIHEnv.render(teleop=teleop, idx=idx)
        ft_cal = get_ft_calibrated()
        PIHEnv.env.viewer_text_overlay(
            text1='F/T Calibrated',
            text2=f'F=[{ft_cal[0]:.2f},{ft_cal[1]:.2f},{ft_cal[2]:.2f}] '
                  f'T=[{ft_cal[3]:.3f},{ft_cal[4]:.3f},{ft_cal[5]:.3f}]',
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    episode_id = 0
    action = np.zeros(7)
    record_flag = False
    phase = 'align'  # 'align' or 'insert'

    # First episode: randomise and prepare alignment
    print(f"\n=== Episode {episode_id + 1}/{NUM_DEMO} ===")
    p_hole = randomise_and_reset()
    init_align(p_hole)
    record_flag = True
    print("Start recording (auto-align phase)")

    while PIHEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
        PIHEnv.step_env()

        if PIHEnv.env.loop_every(HZ=20):

            # ---- Phase 1: auto-align (one step per tick, with recording) ----
            if phase == 'align':
                align_done = step_align()

                # Grab observations & record
                agent_image, wrist_image = PIHEnv.grab_image()
                agent_image = np.array(
                    Image.fromarray(agent_image).resize((256, 256))
                )
                wrist_image = np.array(
                    Image.fromarray(wrist_image).resize((256, 256))
                )
                ee_pose = PIHEnv.get_ee_pose()
                ft = get_ft_calibrated()
                joint_q = PIHEnv.get_joint_state()

                dataset.add_frame(
                    {
                        "observation.image": agent_image,
                        "observation.wrist_image": wrist_image,
                        "observation.state": joint_q[:6],
                        "observation.force_torque": ft,
                        "action": joint_q,
                        "obj_init": PIHEnv.obj_init_pose,
                    },
                    task=PIHEnv.instruction,
                )

                render_with_calibrated_ft(teleop=False, idx=episode_id)

                if align_done:
                    phase = 'insert'
                    print("Switching to teleop insertion phase")
                continue

            # ---- Phase 2: teleop insertion ----
            done = check_success_calibrated()
            if done:
                dataset.save_episode()
                episode_id += 1
                record_flag = False
                print(f"Episode {episode_id}/{NUM_DEMO} saved.")

                if episode_id < NUM_DEMO:
                    print(f"\n=== Episode {episode_id + 1}/{NUM_DEMO} ===")
                    p_hole = randomise_and_reset()
                    init_align(p_hole)
                    phase = 'align'
                    record_flag = True
                    print("Start recording (auto-align phase)")
                continue

            action, reset = PIHEnv.teleop_robot()

            if reset:
                print("Manual reset — re-randomising holes")
                p_hole = randomise_and_reset()
                dataset.clear_episode_buffer()
                init_align(p_hole)
                phase = 'align'
                record_flag = True
                print("Start recording (auto-align phase)")
                continue

            # Grab observations
            agent_image, wrist_image = PIHEnv.grab_image()
            agent_image = np.array(
                Image.fromarray(agent_image).resize((256, 256))
            )
            wrist_image = np.array(
                Image.fromarray(wrist_image).resize((256, 256))
            )

            # Step with action
            joint_q = PIHEnv.step(action)
            action = PIHEnv.q[:7]  # 6 joint angles and 1 gripper
            action = action.astype(np.float32)

            ee_pose = PIHEnv.get_ee_pose()
            ft = get_ft_calibrated()

            dataset.add_frame(
                {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": joint_q[:6],
                    "observation.force_torque": ft,
                    "action": action,
                    "obj_init": PIHEnv.obj_init_pose,
                },
                task=PIHEnv.instruction,
            )

            render_with_calibrated_ft(teleop=True, idx=episode_id)

    PIHEnv.env.close_viewer()

    images_dir = os.path.join(ROOT, 'images')
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)

    print(f"Data collection complete. {episode_id} episodes saved to {ROOT}")


if __name__ == '__main__':
    main()
