"""
pih_gym_wrapper.py — Gymnasium wrapper for PIHEnv2 with SmolVLA integration.

Key design:
  - NO auto-align. SmolVLA base action runs throughout the entire episode.
  - Contact detection: residual RL only activates when peg_tip_site is near
    hole_entry_site (uses 3 MuJoCo sites: peg_tip_site, hole_entry_site, hole_bottom_site).
  - F/T zero calibration at episode start (captures gravity offset ~-13N on z).
  - Before contact: env.step() just forwards SmolVLA action (residual=0 from SAC's perspective,
    but the env doesn't even query the RL policy — the trainer handles that).
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from PIL import Image
from torchvision import transforms

from .config import ResidualRLConfig
from .reward import HierarchicalReward


class PIHResidualEnv(gym.Env):
    """
    Gymnasium environment for residual RL on peg-in-hole insertion.

    Observation:
      - 'state': (15,) = [ft_calibrated(6), ee_pose(6), relative_pos(3)]
      - 'ft_history': (K, 6) = last K calibrated force/torque readings

    Action: (6,) = Cartesian residual [dx, dy, dz, drx, dry, drz].
            The trainer/deployer converts this residual to joint delta before stepping PIHEnv2.
            When contact has not been detected, the trainer should send zeros.

    The wrapper does NOT call SmolVLA internally — the trainer is responsible for:
      1. Calling SmolVLA to get base_action
      2. Calling RL policy to get residual (only after contact)
      3. Passing combined action to env.step()

    This keeps the wrapper simple and gives the trainer full control over
    when to query each policy.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Site names in pih.xml
    PEG_TIP_SITE = 'peg_tip_site'
    HOLE_ENTRY_SITE = 'hole_entry_site'
    HOLE_BOTTOM_SITE = 'hole_bottom_site'

    def __init__(self, pih_env, config=None, render_mode=None):
        """
        Args:
            pih_env:      PIHEnv2 instance (already initialized)
            config:       ResidualRLConfig
            render_mode:  'human' or 'rgb_array'
        """
        super().__init__()
        self.pih = pih_env
        self.cfg = config or ResidualRLConfig()
        self.render_mode = render_mode

        self.reward_fn = HierarchicalReward(self.cfg)

        # Spaces
        self.observation_space = spaces.Dict({
            'state': spaces.Box(-np.inf, np.inf, shape=(self.cfg.state_dim,), dtype=np.float32),
            'ft_history': spaces.Box(-np.inf, np.inf,
                                     shape=(self.cfg.force_history_len, self.cfg.ft_dim),
                                     dtype=np.float32),
        })
        action_low = np.array([
            -self.cfg.max_residual_pos, -self.cfg.max_residual_pos, -self.cfg.max_residual_pos,
            -self.cfg.max_residual_rot, -self.cfg.max_residual_rot, -self.cfg.max_residual_rot,
        ], dtype=np.float32)
        action_high = -action_low
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Internal state
        self._ft_history = deque(maxlen=self.cfg.force_history_len)
        self._step_count = 0
        self._episode_count = 0
        self._ft_offset = np.zeros(6, dtype=np.float32)  # gravity + sensor bias
        self._contact_detected = False

    # ------------------------------------------------------------------
    # F/T calibration
    # ------------------------------------------------------------------
    def _calibrate_ft(self):
        """
        Read current F/T sensor as zero offset. NO extra physics steps.
        Called after pih.reset() which already settles for 100 steps.
        """
        self._ft_offset = self.pih.get_force_torque().copy()

    def get_ft_calibrated(self):
        """Return F/T with zero-offset subtracted (gravity-compensated)."""
        return self.pih.get_force_torque() - self._ft_offset

    # ------------------------------------------------------------------
    # Site position helpers
    # ------------------------------------------------------------------
    def get_site_pos(self, site_name):
        """Get position of a MuJoCo site by name."""
        return self.pih.env.get_p_site(site_name)

    def get_peg_tip_pos(self):
        return self.get_site_pos(self.PEG_TIP_SITE)

    def get_hole_entry_pos(self):
        return self.get_site_pos(self.HOLE_ENTRY_SITE)

    def get_hole_bottom_pos(self):
        return self.get_site_pos(self.HOLE_BOTTOM_SITE)

    # ------------------------------------------------------------------
    # Contact detection
    # ------------------------------------------------------------------
    def check_contact(self):
        """
        Check whether peg has started inserting into hole — residual RL activates.

        Trigger condition: peg_tip is BELOW hole_entry (insertion started),
        AND XY is reasonably close (within 30mm).

        This avoids triggering when peg just touches the hole surface but
        hasn't started actual insertion.

        Once triggered, stays True for the rest of the episode.
        """
        if self._contact_detected:
            return True

        peg_tip = self.get_peg_tip_pos()
        hole_entry = self.get_hole_entry_pos()

        xy_dist = np.linalg.norm(peg_tip[:2] - hole_entry[:2])

        # Peg tip must be BELOW hole entry Z (started entering the hole)
        # AND within 30mm XY of hole center
        peg_below_entry = peg_tip[2] < hole_entry[2]

        if peg_below_entry and xy_dist < 0.03:
            self._contact_detected = True
            ft = self.get_ft_calibrated()
            depth = hole_entry[2] - peg_tip[2]
            print(f"  >> Insertion started: depth={depth*1000:.1f}mm, "
                  f"XY={xy_dist*1000:.1f}mm, Fz={ft[2]:.1f}N")
            return True

        return False

    @property
    def contact_detected(self):
        return self._contact_detected

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset PIHEnv2 — this does:
        #   1. IK to home pose
        #   2. set_instruction() (random peg)
        #   3. 100x step_env() settle
        self.pih.reset(seed=seed)

        # Force round peg. Since pih.reset() already settled with a random peg,
        # we just swap visibility. The peg geometry on the gripper doesn't move,
        # so this doesn't change physics state.
        self.pih.set_instruction('Insert the round peg into the blue round hole.')

        # Calibrate F/T from current sensor reading (NO extra physics steps)
        self._calibrate_ft()

        # Reset internal state
        self._ft_history.clear()
        ft = self.get_ft_calibrated()
        for _ in range(self.cfg.force_history_len):
            self._ft_history.append(ft.copy())

        self.reward_fn.reset(initial_ft=ft)
        self._step_count = 0
        self._episode_count += 1
        self._contact_detected = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action_7d):
        """
        Execute one environment step, matching the original smolvla1.py timing exactly.

        Original loop: step_env(old_q) → if 20Hz: inference → step(new_q)
        So each 25-step block = 1 step with OLD q + 24 steps with NEW q.

        Args:
            action_7d: np.ndarray (7,) — [j1..j6, gripper]

        Returns:
            obs, reward, terminated, truncated, info
        """
        action_7d = np.asarray(action_7d, dtype=np.float32)
        self._step_count += 1

        n_substeps = max(1, int(1.0 / (self.cfg.control_hz * self.pih.env.model.opt.timestep)))

        # Step 1: ONE physics step with OLD self.q (before applying new action)
        # This matches the original: step_env() runs BEFORE loop_every block
        self.pih.step_env()

        # Step 2: Set new joint targets
        self.pih.step(action_7d)

        # Step 3: Run remaining (n-1) physics steps with new action
        for _ in range(n_substeps - 1):
            self.pih.step_env()

        # Update F/T history
        ft = self.get_ft_calibrated()
        self._ft_history.append(ft.copy())

        # Update contact detection
        self.check_contact()

        # Geometry from sites
        peg_tip = self.get_peg_tip_pos()
        hole_entry = self.get_hole_entry_pos()
        hole_bottom = self.get_hole_bottom_pos()

        # Check success
        success = self._check_success()

        # Compute reward (only meaningful after contact; before contact the
        # trainer can choose to ignore it)
        reward, reward_info = self.reward_fn.compute(
            ft=ft,
            peg_tip_site=peg_tip,
            hole_entry_site=hole_entry,
            hole_bottom_site=hole_bottom,
            success=success,
        )

        terminated = success
        truncated = self._step_count >= self.cfg.max_episode_steps

        obs = self._get_obs()
        info = self._get_info()
        info.update(reward_info)
        info['success'] = success
        info['contact_detected'] = self._contact_detected

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Build observation dict."""
        # Calibrated force/torque (6)
        ft = self.get_ft_calibrated()

        # EEF pose (6): [x, y, z, roll, pitch, yaw]
        ee_pose = self.pih.get_ee_pose()

        # Relative position: peg_tip -> hole_entry (3)
        peg_tip = self.get_peg_tip_pos()
        hole_entry = self.get_hole_entry_pos()
        relative = (hole_entry - peg_tip).astype(np.float32)

        # Concatenate: 6 + 6 + 3 = 15
        state = np.concatenate([ft, ee_pose, relative], dtype=np.float32)

        # Force history: (K, 6)
        ft_history = np.array(list(self._ft_history), dtype=np.float32)

        return {
            'state': state,
            'ft_history': ft_history,
        }

    def _get_info(self):
        """Return auxiliary info."""
        peg_tip = self.get_peg_tip_pos()
        hole_entry = self.get_hole_entry_pos()
        hole_bottom = self.get_hole_bottom_pos()
        return {
            'step': self._step_count,
            'episode': self._episode_count,
            'active_peg': self.pih.active_peg,
            'instruction': self.pih.instruction,
            'peg_tip_pos': peg_tip.copy(),
            'hole_entry_pos': hole_entry.copy(),
            'hole_bottom_pos': hole_bottom.copy(),
            'contact_detected': self._contact_detected,
        }

    def _check_success(self):
        """Check insertion success using sites and calibrated F/T."""
        peg_tip = self.get_peg_tip_pos()
        hole_entry = self.get_hole_entry_pos()
        hole_bottom = self.get_hole_bottom_pos()

        # XY alignment
        xy_dist = np.linalg.norm(peg_tip[:2] - hole_entry[:2])
        if xy_dist > 0.03:
            return False

        # Peg must be below hole entry by at least 2cm
        if peg_tip[2] > (hole_entry[2] - 0.02):
            return False

        # Force sanity check (calibrated)
        ft = self.get_ft_calibrated()
        lateral = np.linalg.norm(ft[:2])
        axial = abs(ft[2])
        total_f = np.linalg.norm(ft[:3])
        bending = np.linalg.norm(ft[3:5])
        total_t = np.linalg.norm(ft[3:])

        if (lateral > 20.0 or axial > 80.0 or total_f > 80.0 or
                bending > 1.5 or total_t > 2.0):
            return False
        if axial < 1.0:
            return False

        return True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == 'human':
            self.pih.render(idx=self._episode_count)
        elif self.render_mode == 'rgb_array':
            agent_img, _ = self.pih.grab_image()
            return agent_img

    def close(self):
        self.pih.env.close_viewer()
