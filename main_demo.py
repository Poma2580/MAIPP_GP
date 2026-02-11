# main_demo
from field_env import MultiRobotFieldEnv
from args import get_args
import numpy as np


if __name__ == "__main__":
    # 获取参数
    args = get_args()

    # 创建环境
    env = MultiRobotFieldEnv(args)

    # 重置环境
    obs = env.reset()

    # 简单演示几个步骤
    for step in range(20):
        actions = [env.action_space[i].sample() for i in range(env.num_agents)]
        obs, rewards, done_list, info = env.step(actions)
        print(f"Step {step+1}:")
        print("  Actions:", actions)
        print("  Observations:", obs)
        print("  Rewards:", rewards)
        # print("  Dones:", done_list)
        env.render(block=False)  # ← 新增：渲染当前状态
        env.render_high_info_area(block=True)  # ← 新增：渲染高信息区域
        if any(done_list):
            print("Done!")
            env.render_gp_prediction(block=True)
            break

    # 阻塞保持图形窗口
    env.render(block=True)
    env.render_gp_prediction(block=True)

    # 保存渲染图像
    # env.render(save_path="./demo_render.png")