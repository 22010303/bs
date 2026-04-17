import numpy as np
import time
from mujoco_env.pih_env2 import PIHEnv2


def test_robust_joint_control():
    # 1. 初始化环境 (先测试绝对位置控制)
    action_mode = 'joint_angle'
    env = PIHEnv2(
        xml_path='./asset/pih.xml',
        action_type=action_mode,
        state_type='joint_angle',
    )

    env.reset()
    # 初始状态抓取
    env.grab_image()

    joint_names = [f'joint_{i}' for i in range(6)]
    print(f"=== 开始关节测试 (模式: {action_mode}) ===")

    for test_idx in range(6):
        # --- 准备工作 ---
        env.reset()
        # 获取 reset 后的真实物理状态作为基准
        initial_q = env.get_joint_state()[:6]
        print(f"\n测试关节 [{test_idx}] | 初始角度: {initial_q[test_idx]:.4f}")

        # --- 正向运动测试 ---
        # 核心修复：必须基于当前所有关节的角度来构建 action
        target_q = initial_q.copy()
        target_q[test_idx] += 0.1  # 只改变目标关节

        # 组合成 7 维 action (6关节 + 1夹爪)
        action = np.concatenate([target_q, [200.0]])

        print(f"  -> 指令: 目标关节递增 0.1 弧度")

        # 执行多次 step 以保证控制器有时间到达目标点
        for s in range(300):
            env.step(action)
            env.step_env()  # 物理仿真步进

            if s % 10 == 0:
                env.grab_image()
                env.render(teleop=False)

        # --- 结果校验 (解耦性检查) ---
        final_q = env.get_joint_state()[:6]
        delta = final_q - initial_q

        # 检查目标关节移动量
        print(f"  [结果] 目标关节实际变化: {delta[test_idx]:.4f}")

        # 检查非目标关节的意外漂移 (解耦性)
        other_joints_delta = np.delete(delta, test_idx)
        max_drift = np.max(np.abs(other_joints_delta))
        if max_drift > 1e-3:
            print(f"  [警告] 关节控制存在耦合！非目标关节最大漂移: {max_drift:.4f}")
        else:
            print(f"  [成功] 关节控制完全解耦。")

    print("\n=== 所有关节测试完成 ===")
    env.env.close_viewer()


def test_delta_control_mode():
    """额外测试增量控制模式"""
    print("\n\n=== 开始测试增量控制模式 (delta_joint_angle) ===")
    env = PIHEnv2(
        xml_path='./asset/pih.xml',
        action_type='delta_joint_angle',
        state_type='joint_angle',
    )
    env.reset()

    # 在增量模式下，action 直接就是偏移量
    # 比如 [0.01, 0, 0, 0, 0, 0, 200] 表示 0号关节动 0.01，其他不动
    action = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0])

    q_before = env.get_joint_state()[:6]
    env.step(action)
    env.step_env()
    q_after = env.get_joint_state()[:6]

    print(f"增量前: {q_before[0]:.4f}")
    print(f"增量后: {q_after[0]:.4f}")
    print(f"实际差值: {(q_after[0] - q_before[0]):.4f} (预期接近 0.05)")
    env.env.close_viewer()


if __name__ == "__main__":
    # 建议先运行绝对位置测试，这是最基础的
    test_robust_joint_control()

    # 如果有需要，取消下面注释测试增量模式
    # test_delta_control_mode()