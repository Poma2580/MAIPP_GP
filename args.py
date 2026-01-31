import argparse
import os
def get_args():
    parser = argparse.ArgumentParser(description="Multi-Robot Source Seeking Simulation Parameters")
    # 获取场数据相对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_field_path = os.path.join(current_dir, '..', 'sim_field_data', 'static', 'sources_5','seed_1', 'field_final.npz')
    default_config_path = os.path.join(os.path.dirname(default_field_path), 'config.json')

    # 场参数
    parser.add_argument("--field_path", type=str, default=default_field_path, help="Path to save or load the spatio-temporal field data")

    # 机器人参数
    parser.add_argument("--num_agents", type=int, default=3, help="Number of agents in the simulation")
    parser.add_argument("--config_path", type=str, default=default_config_path, help="Path to config.json containing gp_hyperparams (default: same dir as field_path)")

    # 空间网格参数
    parser.add_argument('--grid_size_x', type=int, default=31)
    parser.add_argument('--grid_size_y', type=int, default=31)
    parser.add_argument('--Lx', type=float, default=30.0)
    parser.add_argument('--Ly', type=float, default=30.0)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--use_relative_pos', action='store_true', default=True)


    # === 高斯过程模型参数 ===
    parser.add_argument('--local_var_len', type=int, default=8,
                        help='Local variance window length for GP local covariance computation')
    parser.add_argument('--gp_grid_N', type=int, default=31,
                        help='Number of grid points per dimension used for Gaussian Process estimation')

    # === 奖励相关系数 ===
    parser.add_argument('--repeat_penalty_coef', type=float, default=3.0,
                        help='Penalty coefficient for revisiting the same cell')
    parser.add_argument('--coordination_coef', type=float, default=0.0,
                        help='Reward coefficient encouraging coordination among agents')
    parser.add_argument('--boundary_coef', type=float, default=3.0,
                        help='Penalty coefficient for approaching boundary or leaving the valid area, ATTENTION: hit boundary will cause the agent to move back')
    parser.add_argument('--conc_reward_coef', type=float, default=0.0,
                        help='Reward coefficient for exploring high concentration regions')
    parser.add_argument('--gp_reward_coef', type=float, default=25.0,
                        help='Reward coefficient for reducing GP predictive uncertainty')

    # === 运动控制参数 ===
    parser.add_argument('--step_size', type=float, default=2.0,
                        help='Step size (movement distance per action)')
    
    # === 测试/日志参数 ===
    parser.add_argument('--number_test', type=int, default=7, help='Run number for logging folder')
    parser.add_argument('--seed_test', type=int, default=0, help='Random seed for testing')
    parser.add_argument('--render', type=int, default=1, help='Render during testing (1 true, 0 false)')
    parser.add_argument('--test_episodes', type=int, default=1, help='Number of test episodes to run')
    parser.add_argument('--model_path', type=str, default=None, help='Explicit path to a model .pth to load during testing')
    parser.add_argument('--step_k', type=int, default=None, help='If specified, load MAPPO_actor_step_{step_k}k.pth from log_dir')
    # === MAPPO 训练与算法超参数 ===
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=20, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=int, default=500, help="Evaluate the policy every N env steps")
    parser.add_argument("--evaluate_times", type=int, default=3, help="How many episodes per evaluation")
    parser.add_argument("--save_model_freq", type=int, default=5000, help="Save the model every N env steps")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (episodes per update)")
    parser.add_argument("--mini_batch_size", type=int, default=16, help="Minibatch size (episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="Hidden size for RNN (if used)")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="Hidden size for MLP")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--K_epochs", type=int, default=15, help="PPO update epochs")

    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick: advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick: reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick: reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=bool, default=False, help="Use ReLU instead of Tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Use RNN policy/value")
    parser.add_argument("--add_agent_id", type=bool, default=True, help="Append agent id to observation")
    parser.add_argument("--use_value_clip", type=bool, default=False, help="Use value clip like PPO2")
    # 统一 env_name 与 number（供日志使用）
    parser.add_argument("--env_name", type=str, default='field_env', help='Environment name for logging')
    parser.add_argument("--number", type=int, default=6, help='Run number for  logging')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # 测试与训练: 若外部需要自定义模型加载, 允许命令行参数生效
    args = parser.parse_args()

    # 合并 grid_size 为 tuple
    args.grid_size = (args.grid_size_y, args.grid_size_x)  # 注意：(Ny, Nx) 对应 field.shape
    


    # print(f"[INFO] Field path: {default_field_path}")
    return args

