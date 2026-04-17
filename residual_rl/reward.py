"""
reward.py — Force-centric reward for residual RL.

Core goal: keep forces SMALL and STABLE during insertion.
SmolVLA already handles the trajectory — residual RL only needs to
make micro-corrections that reduce contact forces.

Reward structure (simple and focused):
  1. Force magnitude penalty: lower force = higher reward
  2. Force stability bonus:   small force change between steps = bonus
  3. Insertion maintenance:    stay in hole and don't retract
  4. Success bonus
"""

import numpy as np
from .config import ResidualRLConfig


class HierarchicalReward:

    def __init__(self, config: ResidualRLConfig = None):
        self.cfg = config or ResidualRLConfig()
        self._prev_insertion_progress = 0.0
        self._max_insertion_progress = 0.0
        self._prev_force = np.zeros(3)
        self._prev_torque = np.zeros(3)
        self._cumulative_force = 0.0
        self._step_count = 0
        self._force_zero = None

    def reset(self, initial_ft=None):
        self._prev_insertion_progress = 0.0
        self._max_insertion_progress = 0.0
        self._prev_force = np.zeros(3)
        self._prev_torque = np.zeros(3)
        self._cumulative_force = 0.0
        self._step_count = 0
        if initial_ft is not None:
            self._force_zero = initial_ft.copy()
        else:
            self._force_zero = np.zeros(6)

    def compute(self, ft, peg_tip_site, hole_entry_site, hole_bottom_site, success):
        cfg = self.cfg
        self._step_count += 1

        # --- Calibrated force ---
        calibrated_ft = ft - self._force_zero
        force = calibrated_ft[:3]
        torque = calibrated_ft[3:]

        lateral_f = np.linalg.norm(force[:2])
        axial_f = abs(force[2])
        total_f = np.linalg.norm(force)
        bending_t = np.linalg.norm(torque[:2])

        # --- Geometry ---
        xy_error = np.linalg.norm(peg_tip_site[:2] - hole_entry_site[:2])
        hole_depth = max(hole_entry_site[2] - hole_bottom_site[2], 0.01)
        insertion_progress = np.clip(
            (hole_entry_site[2] - peg_tip_site[2]) / hole_depth, -1.0, 1.0
        )
        in_hole = (insertion_progress > 0.0) and (xy_error < 0.03)

        reward = 0.0
        info = {}

        # ==============================================================
        # 1. SUCCESS (sparse)
        # ==============================================================
        r_success = cfg.w_success if success else 0.0
        reward += r_success
        info['r_success'] = r_success

        if in_hole:
            # ==========================================================
            # 2. FORCE — per-axis linear penalty above threshold
            #    Fx, Fy: lateral (jamming)
            #    Fz: axial (excessive push)
            #    Tx, Ty: bending torque (skewing)
            # ==========================================================
            r_force_mag = cfg.w_force_baseline

            fx = abs(force[0])
            fy = abs(force[1])
            fz = abs(force[2])
            tx = abs(torque[0])
            ty = abs(torque[1])

            # Per-axis penalties
            p_fx = cfg.penalty_fx * max(0, fx - cfg.fx_th)
            p_fy = cfg.penalty_fy * max(0, fy - cfg.fy_th)
            p_fz = cfg.penalty_fz * max(0, fz - cfg.fz_th)
            p_tx = cfg.penalty_tx * max(0, tx - cfg.tx_th)
            p_ty = cfg.penalty_ty * max(0, ty - cfg.ty_th)

            r_force_mag -= (p_fx + p_fy + p_fz + p_tx + p_ty)

            reward += r_force_mag
            info['r_force_mag'] = r_force_mag
            info['p_fx'] = -p_fx
            info['p_fy'] = -p_fy
            info['p_fz'] = -p_fz
            info['p_tx'] = -p_tx
            info['p_ty'] = -p_ty

            # ==========================================================
            # 3. FORCE STABILITY — reward small force changes
            #    |F_t - F_{t-1}| should be small
            # ==========================================================
            force_change = np.linalg.norm(force - self._prev_force)
            torque_change = np.linalg.norm(torque[:2] - self._prev_torque[:2])

            r_stability = 0.0
            if force_change < cfg.force_stability_threshold:
                # Force is stable — bonus
                r_stability = cfg.w_force_stable
            else:
                # Force jumped — penalty proportional to change
                r_stability = -cfg.penalty_force_jerk * force_change

            reward += r_stability
            info['r_stability'] = r_stability
            info['force_change'] = force_change
            info['torque_change'] = torque_change

            # ==========================================================
            # 4. DEPTH MAINTENANCE — don't retract, keep progressing
            # ==========================================================
            # Small depth-hold bonus
            r_depth = cfg.w_depth_hold * insertion_progress
            reward += r_depth
            info['r_depth'] = r_depth

            # Progress watermark
            r_progress = 0.0
            if insertion_progress > self._max_insertion_progress:
                new_gain = insertion_progress - self._max_insertion_progress
                r_progress = cfg.w_progress * new_gain
                self._max_insertion_progress = insertion_progress
            reward += r_progress
            info['r_progress'] = r_progress

            # Retract penalty
            r_retract = 0.0
            delta = insertion_progress - self._prev_insertion_progress
            if delta < -0.005:
                r_retract = -cfg.penalty_retract * abs(delta)
            reward += r_retract
            info['r_retract'] = r_retract

        else:
            # Outside hole — fixed penalty, no force optimization
            reward -= cfg.penalty_outside
            info['r_force_mag'] = 0.0
            info['p_fx'] = 0.0
            info['p_fy'] = 0.0
            info['p_fz'] = 0.0
            info['p_tx'] = 0.0
            info['p_ty'] = 0.0
            info['r_stability'] = 0.0
            info['r_depth'] = 0.0
            info['r_progress'] = 0.0
            info['r_retract'] = 0.0
            info['force_change'] = 0.0
            info['torque_change'] = 0.0

        # Update state
        self._prev_force = force.copy()
        self._prev_torque = torque.copy()
        self._prev_insertion_progress = insertion_progress
        self._cumulative_force += total_f

        # Info
        info['reward_total'] = reward
        info['in_hole'] = in_hole
        info['insertion_progress'] = insertion_progress
        info['max_progress'] = self._max_insertion_progress
        info['xy_error'] = xy_error
        info['lateral_force'] = lateral_f
        info['axial_force'] = axial_f
        info['total_force'] = total_f
        info['bending_torque'] = bending_t
        info['peg_z'] = peg_tip_site[2]
        info['cumulative_force'] = self._cumulative_force
        info['step_count'] = self._step_count

        return reward, info
