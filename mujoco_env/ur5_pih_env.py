import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

from manipulator_mujoco.arenas import PIHArena
from manipulator_mujoco.robots import Arm, Robotiq_2F85
from manipulator_mujoco.props import PIHPeg
from manipulator_mujoco.controllers import OperationalSpaceController


class UR5ePIHEnv(gym.Env):
    """Single-arm UR5e peg-in-hole insertion environment."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(6,), dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################

        # Arena with table and embedded hole
        self._arena = PIHArena()
        self._arena.mjcf_model.option.gravity = [0, 0, -1]

        # UR5e arm
        self.arm = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site',
            name='arm'
        )

        # Robotiq 2F85 gripper
        self._gripper = Robotiq_2F85()

        # Peg prop (simple cylinder geometry)
        self._peg = PIHPeg()

        # Attach peg RIGIDLY to gripper's pinch site (no hinge joint)
        # pinch site is at z=0.145 on gripper base (between finger pads)
        # Offset the peg so the grip is near the top, peg extends downward
        frame = self._gripper.object_site.attach(self._peg.mjcf_model)
        frame.pos = [0, 0, 0.23]
        frame.quat = [0, 0.7071068, 0.7071068, 0]

        # Attach gripper to arm
        self.arm.attach_tool(
            self._gripper.mjcf_model,
            pos=[0, 0, 0],
            quat=[0, 0, 0, 1]
        )

        # Attach arm to arena - base on the table surface (z=1.05)
        self._arena.attach(
            self.arm.mjcf_model,
            pos=[-0.5, 0, 1.05],
            quat=[0.7071068, 0, 0, -0.7071068]
        )

        # Generate physics
        self.physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # OSC controller
        self._controller = OperationalSpaceController(
            physics=self.physics,
            joints=self.arm.joints,
            eef_site=self.arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=400,
            ko=200,
            kv=30,
            vmax_xyz=3.0,
            vmax_abg=2.0,
        )

        # For GUI and time keeping
        self._timestep = self.physics.model.opt.timestep
        self._viewer = None
        self._step_start = None

    @property
    def gripper(self):
        return self._gripper

    def _get_obs(self) -> np.ndarray:
        return np.zeros(6)

    def _get_info(self) -> dict:
        return {}

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)

        with self.physics.reset_context():
            if options is None:
                # Joint angles for upright table-mounted arm reaching forward and down
                self.physics.bind(self.arm.joints).qpos = [
                    0.0,       # shoulder_pan: facing forward
                    -1.57,     # shoulder_lift: arm horizontal
                    1.57,      # elbow: forearm down
                    -1.57,     # wrist_1: EEF pointing down
                    -1.57,     # wrist_2: tool rotation
                    0.0,       # wrist_3
                ]
            else:
                self.physics.bind(self.arm.joints).qpos = options['arm_pose']

            # Close gripper to hold peg
            self.physics.bind(self._gripper.actuator).ctrl = 200

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        # action is a single 7D pose target [x, y, z, qx, qy, qz, qw]
        self._controller.run(action)
        self.physics.step()

        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()
        return observation, reward, terminated, False, info

    def render(self, camera_id=0) -> np.ndarray:
        if self._render_mode == "rgb_array":
            return self.render_frame(camera_id)

    def render_frame(self, camera_id) -> None:
        if self._viewer is None and self._render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(
                self.physics.model.ptr,
                self.physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            self._step_start = time.time()

        if self._render_mode == "human":
            self._viewer.sync()
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            self._step_start = time.time()
        else:  # rgb_array
            return self.physics.render(960, 1280, camera_id=camera_id)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
