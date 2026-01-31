import torch
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None
import os
import csv
import json
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_network import MAPPO_MPE
from field_env import MultiRobotFieldEnv
from args import get_args
from numpyencoder import NumpyEncoder   # pip install numpyencoder
### nohup python3 /home/ychx/xhk/MAIPP_GP_V2.0/main_training.py > training_output.log 2>&1 &


class Runner_MAPPO_IPP:
    def __init__(self, args):
        self.args = args
        self.env_name = args.env_name
        self.number = args.number
        self.seed = int(args.seed)
        # Seeding
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.args.device = str(self.device)
        print("=========================================================")
        print(f"[Info] Using device: {self.device}")
        print("=========================================================")
        # Env
        self.env = MultiRobotFieldEnv(self.args)
        self.args.N = self.env.n
        self.args.obs_dim_n = [self.env.observation_space[i].shape[0] for i in range(self.args.N)]
        self.args.action_dim_n = [self.env.action_space[i].n for i in range(self.args.N)]
        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.state_dim = int(np.sum(self.args.obs_dim_n))
        print(f"[Env] obs_dim_n={self.args.obs_dim_n}, action_dim_n={self.args.action_dim_n}")
        print(f"[Env] Prior GP variance sum: {self.env.prior_var:.3f}")
        # Agent & buffer
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)
        self.evaluate_rewards = []
        self.total_steps = 0
        if self.args.use_reward_norm:
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)
        # Logging
        self.log_dir = f'./data_train_MPE/{self.env_name}/MAPPO/number_{self.number}_seed_{self.seed}'
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, 'metrics.csv')
        with open(self.log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['total_steps', 'reward_mean', 'reward_std', 'reward_min', 'reward_max'])
        # logging each reward component to new file if needed
        self.reward_components_log_file = os.path.join(self.log_dir, 'reward_components.csv')
        with open(self.reward_components_log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['total_steps', 'conc_reward_mean', 'gp_reward_mean', 'boundary_penalty_mean', 'repeat_penalty_mean', 'coordination_reward_mean'])
        # Save config
        try:
            config = {
                'env_name': self.env_name,
                'number': int(self.number),
                'seed': int(self.seed),
                'device': self.args.device,
                'args': {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, (np.integer,)) else v)
                         for k, v in vars(self.args).items()}
            }
            with open(os.path.join(self.log_dir, 'config.json'), 'w', encoding='utf-8') as cf:
                json.dump(config, cf, cls=NumpyEncoder, ensure_ascii=False, indent=2)
        except Exception as e: 
            print(f"[Warn] failed to write config.json: {e}")

    def run(self):
        evaluate_num = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy(); evaluate_num += 1
            _, ep_steps, _, _, _ = self.run_episode(evaluate=False)
            self.total_steps += ep_steps
            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()
        self.evaluate_policy()

    def evaluate_policy(self):
        rewards = [self.run_episode(evaluate=True)[0] for _ in range(self.args.evaluate_times)]
        traces = [self.run_episode(evaluate=True)[2] for _ in range(self.args.evaluate_times)]
        rmses = [self.run_episode(evaluate=True)[3] for _ in range(self.args.evaluate_times)]
        reward_components = [self.run_episode(evaluate=True)[4] for _ in range(self.args.evaluate_times)]
        if len(rewards) == 0:
            return
        stats = (float(np.mean(rewards)), float(np.std(rewards)), float(np.min(rewards)), float(np.max(rewards)))
        traces_mean = float(np.mean(traces))
        rmses_mean = float(np.mean(rmses))
        with open(self.log_file, 'a', newline='') as f:
            csv.writer(f).writerow([self.total_steps, *stats])
        self.evaluate_rewards.append(stats[0])
        print(f"[Eval] steps={self.total_steps} mean={stats[0]:.3f} std={stats[1]:.3f} min={stats[2]:.3f} max={stats[3]:.3f} posterior_trace_mean={traces_mean:.3f} rmse_mean={rmses_mean:.3f}")

        # log each reward component
        conc_reward_mean = float(np.mean([comp['conc_reward'] for comp in reward_components]))
        gp_reward_mean = float(np.mean([comp['gp_reward'] for comp in reward_components]))
        boundary_penalty_mean = float(np.mean([comp['boundary_penalty'] for comp in reward_components]))
        repeat_penalty_mean = float(np.mean([comp['repeat_penalty'] for comp in reward_components]))
        coordination_reward_mean = float(np.mean([comp['coordination_reward'] for comp in reward_components]))
        with open(self.reward_components_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([self.total_steps, conc_reward_mean, gp_reward_mean, boundary_penalty_mean, repeat_penalty_mean, coordination_reward_mean])
        # Save the model (actor) into the same folder as metrics.csv (uncomment if needed)
        # Example:
        # 间隔保存模型
        if self.total_steps % self.args.save_model_freq == 0:
            self.agent_n.save_model(self.total_steps, save_dir=self.log_dir)

    def run_episode(self, evaluate=False):
        ep_reward = 0.0
        ep_reward_components = {'conc_reward': 0.0,
                              'gp_reward': 0.0,
                              'boundary_penalty': 0.0,
                              'repeat_penalty': 0.0,
                              'coordination_reward': 0.0}
        obs_n = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent_n.reset_rnn_hidden()
        for step in range(self.args.episode_limit):
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            obs_next_n, r_n, done_n, info = self.env.step(a_n)
            ep_reward += float(sum(r_n))
            # accumulate each reward component
            if evaluate:
                ep_reward_components['conc_reward'] += float(sum(info.get('conc_rewards', [0.0]*self.args.N)))
                ep_reward_components['gp_reward'] += float(sum(info.get('gp_rewards', [0.0]*self.args.N)))
                ep_reward_components['boundary_penalty'] += float(sum(info.get('boundary_penalties', [0.0]*self.args.N)))
                ep_reward_components['repeat_penalty'] += float(sum(info.get('repeat_penalties', [0.0]*self.args.N)))
                ep_reward_components['coordination_reward'] += float(sum(info.get('coordination_rewards', [0.0]*self.args.N)))

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)
                self.replay_buffer.store_transition(step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)
            obs_n = obs_next_n
            if all(done_n):
                break
        if not evaluate:
            s_last = np.array(obs_n).flatten()
            v_last = self.agent_n.get_value(s_last)
            self.replay_buffer.store_last_value(step + 1, v_last)
            ep_trace = None
            ep_rmse = None
        # 如果是评估模式，返回协方差矩阵的迹作为指标
        if evaluate:
            # ep_trace = np.sum(self.env.posterior_var)
            ep_trace = None
            ep_rmse = None
            post_mean, post_var, _ = self.env.gp_regressor.predict_on_eval_grid()
            ep_trace = np.sum(post_var)
            ep_rmse = self.env.gp_regressor.compute_rmse(self.env.field, post_mean)
        return ep_reward, step + 1, ep_trace, ep_rmse, ep_reward_components


if __name__ == '__main__':
    args = get_args()
    args.env_name = getattr(args, 'env_name', 'field_env')
    if not hasattr(args, 'episode_limit'):
        args.episode_limit = getattr(args, 'max_steps', 20)
    runner = Runner_MAPPO_IPP(args)
    runner.run()

