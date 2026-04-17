import sys
import random
import numpy as np
import mujoco
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw


class PIHEnv2:
    """
    Language-aware UR5e + Robotiq 2F85 peg-in-hole environment.

    Two holes are always present on the table:
      - hole_A (left)  = square  (hole_1 mesh)
      - hole_B (right) = round   (hole_27 mesh)

    Two peg types exist (only ONE is active per episode):
      - peg_square  (box)      -> must go into hole_A (square)
      - peg_round   (cylinder) -> must go into hole_B (round)

    At reset() the active peg is chosen randomly; the other peg's
    geometry is hidden (alpha=0) and its collision is disabled.
    """

    ARM_JOINTS = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint',
    ]

    EEF_BODY = 'tool0_link'

    # peg_type -> (peg_body, geom_names[], target_hole_body, instruction)
    PEG_CONFIG = {
        'square': {
            'body':   'peg_square',
            'geoms':  ['peg_square_box', 'peg_square_key'],
            'target': 'hole_A',
            'instruction': 'Insert the square peg into the red square hole.',
        },
        'round': {
            'body':   'peg_round',
            'geoms':  ['peg_round_cyl', 'peg_round_key'],
            'target': 'hole_B',
            'instruction': 'Insert the round peg into the blue round hole.',
        },
    }

    # Hole opening (top) z-coordinates after mesh offset + body pos
    # These are where the peg tip must reach for insertion
    HOLE_OPENING_Z = {
        'hole_A': 1.17,  # body z=1.105 + mesh top offset 0.08
        'hole_B': 1.17,  # body z=1.013 + mesh top offset 0.092
    }

    FT_FORCE_SENSOR  = 'force_ee'
    FT_TORQUE_SENSOR = 'torque_ee'

    def __init__(self,
                 xml_path='./asset/pih.xml',
                 action_type='eef_pose',
                 state_type='joint_angle',
                 seed=None):
        self.env = MuJoCoParserClass(name='UR5e_PIH', rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        self.joint_names = self.ARM_JOINTS

        # Cache geom ids for fast toggling
        self._geom_ids = {}
        for ptype, cfg in self.PEG_CONFIG.items():
            for gname in cfg['geoms']:
                gid = mujoco.mj_name2id(
                    self.env.model, mujoco.mjtObj.mjOBJ_GEOM, gname
                )
                self._geom_ids[gname] = gid

        # Store original geom properties for restore
        self._geom_orig = {}
        for gname, gid in self._geom_ids.items():
            self._geom_orig[gname] = {
                'rgba':        self.env.model.geom_rgba[gid].copy(),
                'contype':     int(self.env.model.geom_contype[gid]),
                'conaffinity': int(self.env.model.geom_conaffinity[gid]),
            }

        self.active_peg = None  # set in reset -> set_instruction
        self.init_viewer()
        self.reset(seed)

    # ------------------------------------------------------------------
    # Peg visibility toggling
    # ------------------------------------------------------------------
    def _show_peg(self, peg_type):
        """Restore geom rendering + collision for *peg_type*."""
        for gname in self.PEG_CONFIG[peg_type]['geoms']:
            gid = self._geom_ids[gname]
            orig = self._geom_orig[gname]
            self.env.model.geom_rgba[gid]        = orig['rgba']
            self.env.model.geom_contype[gid]      = orig['contype']
            self.env.model.geom_conaffinity[gid]   = orig['conaffinity']

    def _hide_peg(self, peg_type):
        """Make geom invisible (alpha=0) and disable collision."""
        for gname in self.PEG_CONFIG[peg_type]['geoms']:
            gid = self._geom_ids[gname]
            self.env.model.geom_rgba[gid][3]      = 0.0   # invisible
            self.env.model.geom_contype[gid]       = 0
            self.env.model.geom_conaffinity[gid]   = 0



    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------
    def init_viewer(self):
        self.env.reset()
        self.env.init_viewer(
            distance=2.0,
            elevation=-30,
            transparent=False,
            black_sky=True,
            use_rgb_overlay=False,
            loc_rgb_overlay='top right',
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Home joint config
        q_init = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        q_home, ik_err_stack, ik_info = solve_ik(
            env=self.env,
            joint_names_for_ik=self.joint_names,
            body_name_trgt=self.EEF_BODY,
            q_init=q_init,
            p_trgt=np.array([0.1, 0.0, 1.5]),
            R_trgt=rpy2r(np.deg2rad([180, 0, 0])),
        )
        self.env.forward(q=q_home, joint_names=self.joint_names, increase_tick=False)

        self.last_q = copy.deepcopy(q_home)
        self.q = np.concatenate([q_home, np.array([200.0])])
        self.p0, self.R0 = self.env.get_pR_body(body_name=self.EEF_BODY)

        # Choose peg and configure visibility
        self.set_instruction()

        # Record obj_init: active_peg(3) + hole_A(3) + hole_B(3) = 9
        peg_body = self.PEG_CONFIG[self.active_peg]['body']
        peg_pos = self.env.get_p_body(peg_body)
        hole_A_pos = self.env.get_p_body('hole_A')
        hole_B_pos = self.env.get_p_body('hole_B')
        self.obj_init_pose = np.concatenate(
            [peg_pos, hole_A_pos, hole_B_pos], dtype=np.float32
        )

        # Settle physics
        for _ in range(100):
            self.step_env()

        print("DONE INITIALIZATION")
        self.gripper_state = True
        self.past_chars = []

    # ------------------------------------------------------------------
    # Language instruction
    # ------------------------------------------------------------------
    def set_instruction(self, given=None):
        """
        Randomly pick a peg type (square or round) and set the matching
        instruction + target hole.  Hide the inactive peg.
        """
        if given is None:
            peg_type = random.choice(['square', 'round'])
        else:
            if 'square' in given.lower():
                peg_type = 'square'
            elif 'round' in given.lower():
                peg_type = 'round'
            else:
                raise ValueError(
                    'Instruction must contain "square" or "round".'
                )

        self.active_peg = peg_type
        cfg = self.PEG_CONFIG[peg_type]
        self.instruction = given if given else cfg['instruction']
        self.obj_target  = cfg['target']

        # Show active peg, hide the other
        for pt in self.PEG_CONFIG:
            if pt == peg_type:
                self._show_peg(pt)
            else:
                self._hide_peg(pt)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q, ik_err_stack, ik_info = solve_ik(
                env=self.env,
                joint_names_for_ik=self.joint_names,
                body_name_trgt=self.EEF_BODY,
                q_init=q,
                p_trgt=self.p0,
                R_trgt=self.R0,
                max_ik_tick=50,
                ik_stepsize=1.0,
                ik_eps=1e-2,
                ik_th=np.radians(5.0),
                render=False,
                verbose_warning=False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError(f'Unknown action_type: {self.action_type}')

        gripper_cmd = action[-1]
        self.compute_q = q
        self.q = np.concatenate([q, [gripper_cmd]])

        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            return self.get_delta_q()
        else:
            raise ValueError(f'Unknown state_type: {self.state_type}')

    def step_env(self):
        arm_q = self.q[:6]
        gripper_val = self.q[6] if len(self.q) > 6 else 200.0
        ctrl = np.concatenate([arm_q, [gripper_val]])
        self.env.step(ctrl)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def get_joint_state(self):
        """[j1..j6, gripper_norm] shape (7,)."""
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('right_driver_joint')
        gripper_norm = np.clip(gripper[0] / 0.8, 0.0, 1.0)
        return np.concatenate([qpos, [gripper_norm]], dtype=np.float32)

    def get_ee_pose(self):
        """[px,py,pz,roll,pitch,yaw] shape (6,)."""
        p, R = self.env.get_pR_body(body_name=self.EEF_BODY)
        rpy = r2rpy(R)
        return np.concatenate([p, rpy], dtype=np.float32)

    def get_delta_q(self):
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint('right_driver_joint')
        gripper_norm = np.clip(gripper[0] / 0.8, 0.0, 1.0)
        return np.concatenate([delta, [gripper_norm]], dtype=np.float32)

    def get_force_torque(self):
        """[fx, fy, fz, tx, ty, tz] shape (6,)."""
        force = self.env.get_sensor_value(self.FT_FORCE_SENSOR)
        torque = self.env.get_sensor_value(self.FT_TORQUE_SENSOR)
        return np.concatenate([force, torque], dtype=np.float32)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def grab_image(self):
        self.rgb_agent = self.env.get_fixed_cam_rgb(cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(cam_name='egocentric')
        self.rgb_side = self.env.get_fixed_cam_rgb(cam_name='sideview')
        # # ====== 新增代码：将第一人称视角图像替换为纯黑图像 ======
        # if self.rgb_ego is not None:
        #     # 创建一个与原始图像尺寸、数据类型相同的全零数组（纯黑）
        #     h, w, c = self.rgb_ego.shape
        #     self.rgb_ego = np.zeros((h, w, c), dtype=self.rgb_ego.dtype)
        # # ====== 修改结束 ======
        return self.rgb_agent, self.rgb_ego

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self, teleop=False, idx=0):
        self.env.plot_time()

        p_current, R_current = self.env.get_pR_body(body_name=self.EEF_BODY)
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95, 0.05, 0.05, 0.5])

        rgb_ego_view = add_title_to_img(
            self.rgb_ego, text='Egocentric View', shape=(640, 480)
        )
        rgb_agent_view = add_title_to_img(
            self.rgb_agent, text='Agent View', shape=(640, 480)
        )
        self.env.plot_T(
            p=np.array([0.1, 0.0, 1.0]),
            label=f"Episode {idx}",
            plot_axis=False, plot_sphere=False,
        )
        self.env.viewer_rgb_overlay(rgb_agent_view, loc='top right')
        self.env.viewer_rgb_overlay(rgb_ego_view, loc='bottom right')

        if teleop:
            rgb_side_view = add_title_to_img(
                self.rgb_side, text='Side View', shape=(640, 480)
            )
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(
                text1='Key Pressed',
                text2='%s' % (self.env.get_key_pressed_list()),
            )

        # F/T overlay
        ft = self.get_force_torque()
        self.env.viewer_text_overlay(
            text1='F/T Sensor',
            text2=f'F=[{ft[0]:.2f},{ft[1]:.2f},{ft[2]:.2f}] '
                  f'T=[{ft[3]:.3f},{ft[4]:.3f},{ft[5]:.3f}]',
        )

        # Peg type + instruction overlay
        peg_label = f'Peg: {self.active_peg}'
        self.env.viewer_text_overlay(text1='Active Peg', text2=peg_label)
        if getattr(self, 'instruction', None) is not None:
            self.env.viewer_text_overlay(
                text1='Language Instructions', text2=self.instruction
            )

        # Real-time pose overlay (always visible)
        peg_body = self.PEG_CONFIG[self.active_peg]['body']
        p_peg = self.env.get_p_body(peg_body)
        p_hA = self.env.get_p_body('hole_A')
        p_hB = self.env.get_p_body('hole_B')
        ee = self.get_ee_pose()
        self.env.viewer_text_overlay(
            text1='EEF pose',
            text2=f'[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]',
        )
        self.env.viewer_text_overlay(
            text1=f'Peg ({self.active_peg})',
            text2=f'[{p_peg[0]:.3f}, {p_peg[1]:.3f}, {p_peg[2]:.3f}]',
        )
        self.env.viewer_text_overlay(
            text1='Hole_A (square)',
            text2=f'[{p_hA[0]:.3f}, {p_hA[1]:.3f}, {p_hA[2]:.3f}] top={self.HOLE_OPENING_Z["hole_A"]:.3f}',
        )
        self.env.viewer_text_overlay(
            text1='Hole_B (round)',
            text2=f'[{p_hB[0]:.3f}, {p_hB[1]:.3f}, {p_hB[2]:.3f}] top={self.HOLE_OPENING_Z["hole_B"]:.3f}',
        )

        self.env.render()

    # ------------------------------------------------------------------
    # Teleop
    # ------------------------------------------------------------------
    def teleop_robot(self):
        """
        Keyboard teleoperation.
        Keys: WASD=xy, RF=z, QE=tilt, Arrows=rotation,
              SPACE=toggle gripper, Z=reset,
              P=print poses (debug)
        """
        dpos = np.zeros(3)
        drot = np.eye(3)

        # if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
        #     dpos += np.array([0.005, 0.0, 0.0])
        # if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
        #     dpos += np.array([-0.005, 0.0, 0.0])
        # if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
        #     dpos += np.array([0.0, -0.005, 0.0])
        # if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
        #     dpos += np.array([0.0, 0.005, 0.0])
        # if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
        #     dpos += np.array([0.0, 0.0, 0.005])
        # if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
        #     dpos += np.array([0.0, 0.0, -0.005])
        # 修改这里的数值来调整移动步长
        pos_step = 0.001  # 增加这个值使移动更快，减少则更慢

        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([pos_step, 0.0, 0.0])  # 向前
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-pos_step, 0.0, 0.0])  # 向后
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0, -pos_step, 0.0])  # 向左
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0, pos_step, 0.0])  # 向右
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0, 0.0, pos_step])  # 向上
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0, 0.0, -pos_step])  # 向下

        if self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.03, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.03, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.03, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.03, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.03, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.03, direction=[0.0, 0.0, 1.0])[:3, :3]

        # Reset episode
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True

        # Toggle gripper
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state = not self.gripper_state

        # ---- DEBUG: Press P to print poses ----
        if self.env.is_key_pressed_once(key=glfw.KEY_P):
            self._print_debug_poses()

        # 新增：打印力/力矩指标
        if self.env.is_key_pressed_once(key=glfw.KEY_L):
            self._print_force_metrics()

        drot_rpy = r2rpy(drot)
        gripper_cmd = 200.0 if self.gripper_state else 0.0
        action = np.concatenate(
            [dpos, drot_rpy, np.array([gripper_cmd])], dtype=np.float32
        )
        return action, False

    def _print_debug_poses(self):
        """Print all relevant body poses to terminal for layout debugging."""
        print("\n" + "=" * 60)
        print("  DEBUG POSE DUMP  (press P)")
        print("=" * 60)
        ee = self.get_ee_pose()
        print(f"  EEF (tool0_link): pos=[{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]  "
              f"rpy=[{np.rad2deg(ee[3]):.1f}, {np.rad2deg(ee[4]):.1f}, {np.rad2deg(ee[5]):.1f}]")

        for ptype, cfg in self.PEG_CONFIG.items():
            p = self.env.get_p_body(cfg['body'])
            tag = " <-- ACTIVE" if ptype == self.active_peg else "     (hidden)"
            print(f"  peg_{ptype:6s} ({cfg['body']:12s}): "
                  f"pos=[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]{tag}")

        for name in ['hole_A', 'hole_B']:
            p = self.env.get_p_body(name)
            top_z = self.HOLE_OPENING_Z[name]
            tag = " <-- TARGET" if name == self.obj_target else ""
            print(f"  {name}: pos=[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]  "
                  f"opening_top_z={top_z:.3f}{tag}")

        p_table = self.env.get_p_body('table')
        print(f"  table:  pos=[{p_table[0]:.4f}, {p_table[1]:.4f}, {p_table[2]:.4f}]")

        ft = self.get_force_torque()
        print(f"  F/T: F=[{ft[0]:.3f},{ft[1]:.3f},{ft[2]:.3f}] "
              f"T=[{ft[3]:.4f},{ft[4]:.4f},{ft[5]:.4f}]")
        print(f"  Instruction: {self.instruction}")
        print(f"  Target hole: {self.obj_target}")
        print("=" * 60 + "\n")

    def _print_force_metrics(self):
        """
        简化版的力指标打印
        """
        ft = self.get_force_torque()

        # 计算各种力的指标
        lateral_force = np.sqrt(ft[0] ** 2 + ft[1] ** 2)
        axial_force = abs(ft[2])
        total_force = np.linalg.norm(ft[:3])

        bending_torque = np.sqrt(ft[3] ** 2 + ft[4] ** 2)
        twist_torque = abs(ft[5])
        total_torque = np.linalg.norm(ft[3:])

        print("\n" + "=" * 50)
        print("力/力矩指标:")
        print("=" * 50)
        print(f"侧向力 (XY): {lateral_force:7.2f} N")
        print(f"轴向力 (Z):  {axial_force:7.2f} N")
        print(f"总力:        {total_force:7.2f} N")
        print(f"弯曲力矩:    {bending_torque:7.3f} N·m")
        print(f"扭转力矩:    {twist_torque:7.3f} N·m")
        print(f"总力矩:      {total_torque:7.3f} N·m")
        print("=" * 50)

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------
    def check_success(self):
        """
        Peg is inserted when:
          - peg xy is close to hole xy
          - peg z is BELOW the hole opening (peg has entered the hole)
        """
        peg_body = self.PEG_CONFIG[self.active_peg]['body']
        p_peg = self.env.get_p_body(peg_body)
        p_hole = self.env.get_p_body(self.obj_target)
        hole_top_z = self.HOLE_OPENING_Z[self.obj_target]

        xy_dist = np.linalg.norm(p_peg[:2] - p_hole[:2])
        # Peg must be below the hole opening by at least 2cm
        peg_below_opening = p_peg[2] < (hole_top_z - 0.02)

        if not (xy_dist < 0.03 and peg_below_opening):
            return False  # 位置不满足

        # 2. 力检查（使用模计算，处理正负值）
        ft = self.get_force_torque()  # [fx, fy, fz, tx, ty, tz]

        # 计算各种力的指标
        lateral_force = np.sqrt(ft[0] ** 2 + ft[1] ** 2)  # XY平面力大小
        axial_force = abs(ft[2])  # Z方向力大小（绝对值）
        total_force = np.linalg.norm(ft[:3])  # 总力大小

        bending_torque = np.sqrt(ft[3] ** 2 + ft[4] ** 2)  # 弯曲力矩大小
        twist_torque = abs(ft[5])  # 扭转力矩大小
        total_torque = np.linalg.norm(ft[3:])  # 总力矩大小

        # 3. 阈值设置（根据您的系统调整）
        thresholds = {
            'lateral_force': 20.0,  # 侧向力不应太大
            'axial_force_max': 80.0,  # 最大轴向压力
            'total_force_max': 80.0,  # 最大总力
            'bending_torque': 1.5,  # 最大弯曲力矩
            'twist_torque': 1.5,  # 最大扭转力矩
            'total_torque': 2,  # 最大总力矩
        }

        # 4. 检查是否超过阈值
        if (lateral_force > thresholds['lateral_force'] or
                axial_force > thresholds['axial_force_max'] or
                total_force > thresholds['total_force_max'] or
                bending_torque > thresholds['bending_torque'] or
                twist_torque > thresholds['twist_torque'] or
                total_torque > thresholds['total_torque']):
            return False  # 力过大，可能是卡住

        # 5. 可选：检查是否有最小接触力
        # 如果轴向力太小，可能还没真正接触
        if axial_force < 1.0:  # 至少2N的接触力
            return False

        # 所有条件满足
        return True

    def check_success_with_debug(self):
        result = self.check_success()

        if not result:
            # 获取详细信息用于调试
            peg_body = self.PEG_CONFIG[self.active_peg]['body']
            p_peg = self.env.get_p_body(peg_body)
            p_hole = self.env.get_p_body(self.obj_target)
            hole_top_z = self.HOLE_OPENING_Z[self.obj_target]

            xy_dist = np.linalg.norm(p_peg[:2] - p_hole[:2])
            insertion_depth = max(0, hole_top_z - p_peg[2])

            ft = self.get_force_torque()

            print(f"失败原因分析:")
            print(f"  XY误差: {xy_dist * 1000:.1f}mm {'✓' if xy_dist < 0.03 else '✗'}")
            print(f"  插入深度: {insertion_depth * 1000:.1f}mm")
            print(f"  力: F=[{ft[0]:.1f}, {ft[1]:.1f}, {ft[2]:.1f}] N")
            print(f"  力矩: T=[{ft[3]:.3f}, {ft[4]:.3f}, {ft[5]:.3f}] N·m")

        return result

    # ------------------------------------------------------------------
    # Object poses
    # ------------------------------------------------------------------
    def get_obj_pose(self):
        """Return positions of active peg, hole_A, hole_B."""
        peg_body = self.PEG_CONFIG[self.active_peg]['body']
        p_peg = self.env.get_p_body(peg_body)
        p_hole_A = self.env.get_p_body('hole_A')
        p_hole_B = self.env.get_p_body('hole_B')
        return p_peg, p_hole_A, p_hole_B
