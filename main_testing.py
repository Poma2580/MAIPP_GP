import os
import re
import torch
import numpy as np
from args import get_args
from field_env import MultiRobotFieldEnv
from mappo_network import MAPPO_MPE

"""
Testing script for MAPPO policy.
Usage: python main_testing.py --env_name field_env --number 0 --seed 0 --episode_limit 50 --test_episodes 10
It will:
  1. Locate the log_dir: ./data_train_MPE/{env_name}/MAPPO/number_{number}_seed_{seed}
  2. Auto-select the latest checkpoint (highest step_*k pth) or use --model_path to specify one.
  3. Load the actor network and run test episodes with rendering.
  4. Print per-episode reward statistics.
"""

def find_latest_checkpoint(log_dir: str):
    pattern = re.compile(r"MAPPO_actor_step_(\d+)k\.pth")
    latest_step = -1
    latest_path = None
    if not os.path.isdir(log_dir):
        return None
    for fname in os.listdir(log_dir):
        m = pattern.match(fname)
        if m:
            step_k = int(m.group(1))
            if step_k > latest_step:
                latest_step = step_k
                latest_path = os.path.join(log_dir, fname)
    return latest_path

def load_model(actor, path, device):
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"[Error] 模型文件未找到: {path}")
    state_dict = torch.load(path, map_location=device)
    actor.load_state_dict(state_dict)
    print(f"[Load] 成功载入模型: {os.path.basename(path)}")


def evaluate(env, agent: MAPPO_MPE, episodes: int, render: bool = True):
    rewards = []
    traces = []
    rmses = []
    for epi in range(episodes):
        obs_n = env.reset()
        if agent.use_rnn:
            agent.actor.rnn_hidden = None  # type: ignore[assignment]
            agent.critic.rnn_hidden = None  # type: ignore[assignment]
        ep_reward = 0.0
        ep_trace = 0.0
        for step in range(agent.episode_limit):
            a_n, _ = agent.choose_action(obs_n, evaluate=True)
            obs_n, r_n, done_n, info = env.step(a_n)
            ep_reward += float(sum(r_n))
            # if render:
            #     try:
            #         env.render(block=False)
                # except Exception:
                #     pass
            if all(done_n):
                break
        # ep_trace = np.sum(env.posterior_var)
        ep_trace = None
        ep_rmse = None
        post_mean, post_var, _ = env.gp_regressor.predict_on_eval_grid()
        ep_trace = np.sum(post_var)
        ep_rmse = env.gp_regressor.compute_rmse(env.field, post_mean)

        rewards.append(ep_reward)
        traces.append(ep_trace)
        rmses.append(ep_rmse)
        print(f"[Episode {epi+1}] reward={ep_reward:.3f} posterior_trace={ep_trace:.3f} rmse={ep_rmse:.3f}")
        # ⚠ 这一行改成 block=True：每个 episode 结束弹一次、等你关窗口
        if render:
            env.render(block=True)
            env.render_gp_prediction(block=True)
    rewards = np.array(rewards, dtype=np.float32)
    print("=========================================================")
    print(f"[Summary] episodes={episodes} mean={rewards.mean():.3f} std={rewards.std():.3f} min={rewards.min():.3f} max={rewards.max():.3f}")
    print("=========================================================")


def main():
    args = get_args()
    args.env_name = getattr(args, 'env_name', 'field_env')
    number = getattr(args, 'number_test', 0)
    seed = int(getattr(args, 'seed_test', 0))
    test_episodes = getattr(args, 'test_episodes', 1)

    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.device = device
    print("=========================================================")
    print(f"[Info] Testing on device: {device}")
    print("=========================================================")

    # build env (needs dimensions for agent init)
    env = MultiRobotFieldEnv(args)
    args.N = env.n
    args.obs_dim_n = [env.observation_space[i].shape[0] for i in range(args.N)]
    args.action_dim_n = [env.action_space[i].n for i in range(args.N)]
    args.obs_dim = args.obs_dim_n[0]
    args.action_dim = args.action_dim_n[0]
    args.state_dim = int(np.sum(args.obs_dim_n))
    print(f"[Env] Prior GP variance sum: {env.prior_var:.3f}")

    agent = MAPPO_MPE(args)

    # log directory
    log_dir = f'./data_train_MPE/{args.env_name}/MAPPO/number_{number}_seed_{seed}'
    ckpt_path = getattr(args, 'model_path', None)
    # step_k 指定时构造文件名
    if ckpt_path is None:
        step_k = getattr(args, 'step_k', None)
        if step_k is not None:
            candidate = os.path.join(log_dir, f"MAPPO_actor_step_{int(step_k)}k.pth")
            if os.path.isfile(candidate):
                ckpt_path = candidate
            else:
                print(f"[Warn] 指定 step_k={step_k} 的文件不存在: {candidate}")
        # 若未找到或未指定 step_k，回退最新
        if ckpt_path is None:
            ckpt_path = find_latest_checkpoint(log_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"未找到任何 checkpoint，请确认训练已生成模型文件，目录: {log_dir}")

    load_model(agent.actor, ckpt_path, device)

    evaluate(env, agent, episodes=test_episodes, render=bool(getattr(args, 'render', 1)))


if __name__ == '__main__':
    main()
