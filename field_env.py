# field_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import torch
from GP_utils import GPExactRegressor

class MultiRobotFieldEnv:
    def __init__(self, args, reward_fn=None, done_fn=None):
        self.args = args
        self.num_agents = args.num_agents
        # 每个位置周围选取的高价值区域点个数（用于构造固定长度观测）
        self.high_info_K = int(getattr(args, 'high_info_K', 8))
        # 对齐外部训练器的接口
        self.n = self.num_agents
        self.grid_size = args.grid_size  # (Ny, Nx)
        self.Lx, self.Ly = args.Lx, args.Ly
        self.dx = self.Lx / (self.grid_size[1] - 1)  # Nx = grid_size[1]
        self.dy = self.Ly / (self.grid_size[0] - 1)  # Ny = grid_size[0]
        x_coords = np.linspace(0, self.Lx, self.grid_size[1])
        y_coords = np.linspace(0, self.Ly, self.grid_size[0])
        # 按坐标点排列: [x1, y1], [x1, y2], [x1, y3], ...
        self.map_coords = [np.array([x, y], dtype=np.float32) for x in x_coords for y in y_coords]
        self.map_values, self.map_vars = None, None  # GP 预测的均值/方差网格（用于可视化/调试等）
        self.max_steps = args.max_steps
        self.use_relative_pos = args.use_relative_pos
        self.np_random = np.random.RandomState(args.seed)
        self.local_var_len = getattr(args, 'local_var_len', 4) # Todo: could be parameterized
        self.gp_grid_N = getattr(args, 'gp_grid_N', 10)
        # 重复访问惩罚，正数表示惩罚幅度，实际奖励中作为减项
        self.repeat_penalty_coef = float(getattr(args, 'repeat_penalty_coef', 1.0))
        # 协调奖励系数
        self.coordination_coef = float(getattr(args, 'coordination_coef', 0.2))
        # 边界惩罚系数
        self.boundary_coef = float(getattr(args, 'boundary_coef', 0.5))
        # 浓度奖励系数
        self.conc_reward_coef = float(getattr(args, 'conc_reward_coef', 0.5))
        # 高斯过程奖励系数
        self.gp_reward_coef = float(getattr(args, 'gp_reward_coef', 0.5))
        # 加载场
        data = np.load(args.field_path)
        self.field = data['z']
        assert self.field.shape == self.grid_size, f"Field shape {self.field.shape} != {self.grid_size}"

        # 加载高斯过程超参数（如果需要）
        self.gp_lengthscale_factor = getattr(args, 'gp_lengthscale_factor', 3.0)
        self.conc_max = None
        self.conc_min = 0.0
        if args.config_path:
            with open(args.config_path, 'r') as f:
                config = json.load(f)
                # convenience attributes
                self.gp_hyperparams = config.get('gp_hyperparams', {})
                # 直接从 gp_hyperparams 中获取参数，
                self.gp_lengthscale = float(self.gp_hyperparams.get('lengthscale', np.nan)) * self.gp_lengthscale_factor
                self.gp_outputscale = float(self.gp_hyperparams.get('outputscale', np.nan))  
                self.gp_noise = float(self.gp_hyperparams.get('noise', np.nan)) 
                self.gp_mode = self.gp_hyperparams.get('mode', None)  # 'exact' or 'svgp' if saved
                self.gp_hyperparams['lengthscale'] = self.gp_lengthscale
                # --- 提取 sources 中的 amp 字段作为 conc_max 浓度最大值(选择最大的 amp) ---
                sources = config.get('sources', None)
                if sources and isinstance(sources, list):
                    amps = []
                    for s in sources:
                        try:
                            a = float(s.get('amp', np.nan))
                            if not np.isnan(a):
                                amps.append(a)
                        except Exception:
                            continue
                    if len(amps) > 0:
                        self.conc_max = float(max(amps))
                # -- 定义基准协方差 --
                self.sigma_f = self.gp_outputscale if self.gp_outputscale is not None else 1.0
        else:
            # 确保存在 gp_hyperparams 字段
            self.gp_hyperparams = {}

        # 若未从配置中得到 conc_max，回退为场数据的最大值
        if self.conc_max is None:
            try:
                self.conc_max = float(np.max(self.field))
            except Exception:
                self.conc_max = 1.0

        # 归一化场值
        self.field = (self.field - self.conc_min) / (self.conc_max - self.conc_min)
        
        # 归一化高斯超参
        self.gp_outputscale /= (self.conc_max - self.conc_min) ** 2
        self.gp_noise /= (self.conc_max - self.conc_min) ** 2

        # 设定初始高斯协方差之和
        self.prior_var =  self.gp_grid_N * self.gp_grid_N * self.sigma_f 

        # UCB 探索参数
        self.ucb_beta = args.ucb_beta
        self.ucb_threshold = args.ucb_threshold

        # 动作：8方向（dy, dx）
        self.step_size = getattr(args, 'step_size', 4)  # Todo: could be parameterized
        # 对齐主训练循环接口：每个 agent 一个离散动作空间
        self.action_space = [spaces.Discrete(8) for _ in range(self.num_agents)]
        self.action_to_delta =  [
            (-1, 0),   # ↑
            (-1, 1),   # ↗
            (0, 1),    # →
            (1, 1),    # ↘
            (1, 0),    # ↓
            (1, -1),   # ↙
            (0, -1),   # ←
            (-1, -1)   # ↖
        ]

        # 计算 observation_space 维度并创建 gym space（float）
        # base obs per agent:
        # [pos_x, pos_y, self_nearby_means(K), self_nearby_vars(K)]
        base_dim = 2 + 2 * self.high_info_K
        # other agents messages per neighbor:
        # [agent_id, rel_x, rel_y, neighbor_nearby_mean_avg, neighbor_nearby_var_avg]
        neighbor_dim = (self.num_agents - 1) * 5
        total_dim = base_dim + neighbor_dim
        # 对齐主训练循环接口：为每个 agent 提供同构的观测空间
        self._single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
        self.observation_space = [self._single_observation_space for _ in range(self.num_agents)]


        # 回调函数
        self.reward_fn = reward_fn if reward_fn else self._default_reward
 

        # 图像
        self.trajectories = [[] for _ in range(self.num_agents)]
        # 测量序列：与 trajectories 一一对应，仅记录 z 值，位置仍使用 trajectories 中的 (px, py)
        self.meas_values = [[] for _ in range(self.num_agents)]
        self.fig = None
        self.ax = None


        # internal state containers 内部状态定义
        self.positions = [(0, 0) for _ in range(self.num_agents)]
        self.step_count = 0
        self.time_left = self.max_steps
        # last_action: store as (dx, dy) in meters units (float)
        self.last_actions = [(0.0, 0.0) for _ in range(self.num_agents)]
        # z history per agent: keep last 2 measurements
        self.z_history = [ [0.0, 0.0] for _ in range(self.num_agents) ]  # list of lists (oldest first)
        # local_var placeholders per agent (vector of length local_var_len)
        self.local_vars_state = [ np.zeros(self.local_var_len, dtype=np.float32) for _ in range(self.num_agents) ]
        
        # 最近一步是否重复标志（便于外部查看/调试）
        self.last_repeat_flags = [False for _ in range(self.num_agents)]


        # self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        # 随机初始化位置（避免重叠）
        positions = []
        while len(positions) < self.num_agents:
            pos = (
                self.np_random.randint(0, self.grid_size[0]),
                self.np_random.randint(0, self.grid_size[1])
            )
            if pos not in positions:
                positions.append(pos)
        # 改为固定初始位置，便于调试
        # positions = [(7, 8), (2, self.grid_size[1]-7), (self.grid_size[0]-6, 11)]
        self.positions = positions
        self.step_count = 0
        self.time_left = self.max_steps
        # 重置最近一步重复标志
        self.last_repeat_flags = [False for _ in range(self.num_agents)]

        # 清空轨迹并记录初始位置
        self.trajectories = [[] for _ in range(self.num_agents)]
        self.meas_values = [[] for _ in range(self.num_agents)]
        for i, (y, x) in enumerate(self.positions):
            px = x * self.dx
            py = y * self.dy
            self.trajectories[i].append((px, py))
   
        # 根据初始位置定义历史动作和历史浓度
        for i, (y, x) in enumerate(self.positions):
            conc = float(self.field[y, x] + self.np_random.normal(0, 0.01))
            # z_history keep last 2 values (oldest first)
            self.z_history[i] = [conc, conc]
            self.last_actions[i] = (0.0, 0.0)
            # 同步记录初始测量值（与 trajectories 对齐）
            self.meas_values[i].append(conc)

        # 根据初始位置计算初始化高斯函数
        payload = self.train_pos()
        self.gp_regressor = GPExactRegressor(payload, gp_params=self.gp_hyperparams, grid_N=self.gp_grid_N)
        # 获取高价值区域
        self.high_info_area = self.gp_regressor.get_high_info_area(threshold=self.ucb_threshold, beta=self.ucb_beta)

        # 计算各种指标
        self.cov_trace = self.gp_regressor.compute_cov_trace(self.high_info_area)
        self.rmse = self.gp_regressor.compute_rmse(self.field)
        self.MI = self.gp_regressor.compute_mutual_info(self.high_info_area)
        # 计算初始全局高斯均值与方差 shape: np.array (Nx, Ny)
        self.map_values, self.map_vars, _ = self.gp_regressor.predict_on_eval_grid()

        obs = self._get_obs(self.high_info_area)
        # 与主训练循环对齐：仅返回 obs 列表
        return obs

    def _get_obs(self, high_info_area=None):
        """
        返回每个agent的观测列表list, 每个观测为np.array(float32)
        布局：
        [pos_x, pos_y, 周围高价值区域点的均值(K)和方差(K),
         then for each other agent j != i: [agent_id, rel_x, rel_y, 每个邻居周围高价值点的均值/方差各自取平均] ]
        并且做归一化处理（位置/相对位置缩放到 [0,1] 左右，agent_id 归一到 [0,1]）
        """
        obs_list = []
        # 构建观测
        for i in range(self.num_agents):
            y, x = self.positions[i]
            # 自身真实位置（以长度单位）
            x_pos = x * self.dx
            y_pos = y * self.dy
            # 观测基础部分：自身位置
            base = [float(x_pos), float(y_pos)]

            # 观测部分：自身周围高价值区域点的均值和方差（固定长度 K）
            if high_info_area is not None and len(high_info_area) > 0:
                nearby_means, nearby_vars = self.gp_regressor.get_nearby_high_info(
                    query_points=np.array([[x_pos, y_pos]], dtype=np.float32),
                    high_info_area=high_info_area,
                    K=self.high_info_K,
                    agent_id=None, 
                )
                # get_nearby_high_info 对单个 query 会返回 shape (K,)
                nearby_means = np.asarray(nearby_means, dtype=np.float32).reshape(-1)
                nearby_vars = np.asarray(nearby_vars, dtype=np.float32).reshape(-1)
                if nearby_means.shape[0] < self.high_info_K:
                    nearby_means = np.pad(nearby_means, (0, self.high_info_K - nearby_means.shape[0]))
                if nearby_vars.shape[0] < self.high_info_K:
                    nearby_vars = np.pad(nearby_vars, (0, self.high_info_K - nearby_vars.shape[0]))
            else:
                nearby_means = np.zeros((self.high_info_K,), dtype=np.float32)
                nearby_vars = np.zeros((self.high_info_K,), dtype=np.float32)

            base.extend(nearby_means.tolist())
            base.extend(nearby_vars.tolist())

            # 邻居信息构建 for i: fixed order j=0..num_agents-1, skip i
            for j in range(self.num_agents):
                if j == i:
                    continue
                # agent id as scalar (could be one-hot elsewhere)
                agent_id = float(j)
                yj, xj = self.positions[j]
                rel_x = float((xj - x) * self.dx)
                rel_y = float((yj - y) * self.dy)

                # 邻居周围高价值区域点的 mean/var 各自取平均（固定 2 维）
                if high_info_area is not None and len(high_info_area) > 0:
                    n_x_pos = float(xj * self.dx)
                    n_y_pos = float(yj * self.dy)
                    n_means, n_vars = self.gp_regressor.get_nearby_high_info(
                        query_points=np.array([[n_x_pos, n_y_pos]], dtype=np.float32),
                        high_info_area=high_info_area,
                        K=self.high_info_K,
                        agent_id=None,  
                    )
                    n_means = np.asarray(n_means, dtype=np.float32).reshape(-1)
                    n_vars = np.asarray(n_vars, dtype=np.float32).reshape(-1)
                    neighbor_mean_avg = float(np.mean(n_means)) if n_means.size > 0 else 0.0
                    neighbor_var_avg = float(np.mean(n_vars)) if n_vars.size > 0 else 0.0
                else:
                    neighbor_mean_avg = 0.0
                    neighbor_var_avg = 0.0

                base.extend([agent_id, rel_x, rel_y, neighbor_mean_avg, neighbor_var_avg])

            obs_array = np.array(base, dtype=np.float32)
            # 归一化 obs 向量
            obs_norm = self._normalize_obs_vector(obs_array)
            obs_list.append(obs_norm)
        return obs_list
    
    def step(self, actions):
        """
        actions: list of int, length should be self.num_agents
        returns: obs_list, rewards_list, terminated_list, truncated_list, info
        """
        assert len(actions) == self.num_agents, "Invalid actions length"
        self.step_count += 1

        # 更新位置（边界反弹+惩罚）
        new_positions = []
        hit_boundaries = [False for _ in range(self.num_agents)]
        repeat_flags = [False for _ in range(self.num_agents)]


        for i, action in enumerate(actions):
            dy, dx = self.action_to_delta[action]
            y, x = self.positions[i]
            desired_y = y + dy * self.step_size
            desired_x = x + dx * self.step_size

            # clip to grid and set flag if clipped
            ny = int(np.clip(desired_y, 0, self.grid_size[0] - 1))
            nx = int(np.clip(desired_x, 0, self.grid_size[1] - 1))

            if (ny != desired_y) or (nx != desired_x):
                hit_boundaries[i] = True
                # reflect last action (set to opposite movement in meters)
                self.last_actions[i] = ( -dx * self.dx, -dy * self.dy )
            else:
                # normal move: set last_action to actual movement in meters
                self.last_actions[i] = ( dx * self.dx, dy * self.dy )

            # 是否走入重复格子：基于已有轨迹 self.trajectories[i]（存储的是米制坐标），
            # 将其映射回网格索引后判断 (ny, nx) 是否已出现过
            if len(self.trajectories[i]) > 0:
                seen_cells = set()
                for (px_prev, py_prev) in self.trajectories[i]:
                    y_prev = int(round(py_prev / self.dy))
                    x_prev = int(round(px_prev / self.dx))
                    seen_cells.add((y_prev, x_prev))
                if (ny, nx) in seen_cells:
                    repeat_flags[i] = True

            new_positions.append((int(ny), int(nx)))

        self.positions = new_positions
        # 更新最近一步的重复标志到环境状态，便于外部查看
        self.last_repeat_flags = repeat_flags

        # 记录新位置到轨迹（米制坐标）
        for i, (y, x) in enumerate(self.positions):
            px = x * self.dx
            py = y * self.dy
            self.trajectories[i].append((px, py))

        # 更新 z_history based on new positions (with small sensor noise)
        for i, (y, x) in enumerate(self.positions):
            z_val = float(self.field[y, x] + self.np_random.normal(0, 0.01))
            # append to history, keep last 2
            hist = self.z_history[i]
            hist.append(z_val)
            if len(hist) > 2:
                hist.pop(0)
            self.z_history[i] = hist
            # 同步记录测量值到 meas_values（与 trajectories 对齐）
            self.meas_values[i].append(z_val)


        # 更新高斯过程的训练数据
        payload = self.train_pos()
        self.gp_regressor.update_dataset_from_payload(payload)
        # 更新地图高斯均值与方差
        self.map_values, self.map_vars, _ = self.gp_regressor.predict_on_eval_grid()
        # 更新高价值区域
        self.high_info_area = self.gp_regressor.get_high_info_area(threshold=0.5, beta=0.5)
        # 打印高价值区域个数
        # print(f"High info area size: {len(self.high_info_area)}")
        # 计算各种指标
        self.cov_trace_new = self.gp_regressor.compute_cov_trace(self.high_info_area)
        self.rmse = self.gp_regressor.compute_rmse(self.field)
        self.MI = self.gp_regressor.compute_mutual_info(self.high_info_area)
        
        # 更新观测（把当前 high_info_area 作为输入，保持观测维度一致）
        obs = self._get_obs(self.high_info_area)

        # 计算奖励（每个智能体）与奖励组成部分
        rewards = []
        conc_rewards = []
        gp_rewards = []
        boundary_penalties = []
        repeat_penalties = []
        coordination_rewards = []

        for i in range(self.num_agents):
            r, conc_r, gp_r, bound_pen, repeat_pen, coor_pen = self.reward_fn(i, actions[i], hit_boundaries[i])
            rewards.append(r)
            conc_rewards.append(conc_r)
            gp_rewards.append(gp_r)
            boundary_penalties.append(bound_pen)
            repeat_penalties.append(repeat_pen)
            coordination_rewards.append(coor_pen)

        # 把奖励组成部分变成 info 方便外部查看/记录
        info = {
            'hit_boundaries': hit_boundaries,
            'repeat_flags': [int(f) for f in repeat_flags],
            'conc_rewards': conc_rewards,
            'gp_rewards': gp_rewards,
            'boundary_penalties': boundary_penalties,
            'repeat_penalties': repeat_penalties,
            'coordination_rewards': coordination_rewards
        }
        # 更新计时器与终止条件（done when max_steps reached）
        self.time_left = max(0, self.max_steps - self.step_count)
        terminated = (self.step_count >= self.max_steps)
        terminated_list = [terminated] * self.num_agents
        truncated_list = [False] * self.num_agents

        # 更新协方差迹
        self.cov_trace = self.cov_trace_new


        # 与主训练循环对齐：返回(obs, rewards, done_list, info)
        done_list = [bool(t or tr) for t, tr in zip(terminated_list, truncated_list)]
        return obs, rewards, done_list, info

    # 默认奖励：浓度越高越好
    def _default_reward(self, agent_id, action, hit_boundary):
        """
        默认奖励函数：
        组成：
        - 浓度奖励：当前位置场值的 min-max 归一化
        - 边界惩罚：接近边界时给予惩罚
        - 协调奖励：基于与其他智能体的相对位置计算奖励，鼓励合理分布
        - 重复路径惩罚：如果当前位置在历史轨迹中出现过，给予惩罚
        - 高斯过程不确定性奖励：基于 GP 预测不确定性的减少量给予奖励
        Parameters
        """
        # 当前（已移动后的）位置，用于重复判定
        y_i, x_i = self.positions[agent_id]


        # 基础奖励：场值(min_max 归一化)
        conc_reward = float(self.field[y_i, x_i])

        # 边界惩罚 penalty scale [0, 1]
        bound_penalty = -1.0 if hit_boundary else 0.0

        # ===== 改为“距离过近惩罚”版协调奖励 =====
        # 参数（可根据实验调优）
        safe_dist = 7.0           # 安全距离阈值（以格子或同坐标系单位），小于该值被惩罚
        penalty_scale = 1.0       # 惩罚强度缩放（越大惩罚越重）
        max_penalty_per_neighbor = 1.0  # 单个邻居最大惩罚（用于归一化或截断）
        eps = 1e-6                # 数值稳定项
        penalties = []

        for j in range(self.num_agents):
            if j == agent_id:
                continue
            y_j, x_j = self.positions[j]

            # 计算欧氏距离（格子单位）
            d = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)

            # 若距离小于安全阈值，给出平滑惩罚；否则惩罚为 0
            if d < safe_dist:
                # 用二次项做平滑惩罚（靠得越近，惩罚越大）
                frac = max(0.0, 1.0 - (d / (safe_dist + eps)))
                penalty = penalty_scale * (frac ** 2)

                # 截断到单个邻居最大惩罚，防止单邻居惩罚爆炸
                penalty = min(penalty, max_penalty_per_neighbor)
            else:
                penalty = 0.0

            penalties.append(penalty)

        # 汇总惩罚：取平均（也可以取和，根据你 reward scale 决定）
        if penalties:
            coor_pen = - float(np.mean(penalties))  # 负值表示惩罚
        else:
            coor_pen = 0.0

        # 重复路径惩罚：检查当前实际格子是否在历史轨迹（含本步）中
        repeat_pen = -1.0 if self.last_repeat_flags[agent_id] else 0.0

        # 高斯过程不确定性奖励（for agent_id）

        delta_I =  (self.cov_trace - self.cov_trace_new) / self.cov_trace  # 归一化的信息增益
        gp_reward = float(delta_I) if delta_I > 0 else 0.0

        # 总奖励
        total_reward = conc_reward * self.conc_reward_coef +  \
                    coor_pen * self.coordination_coef + \
                    bound_penalty * self.boundary_coef + \
                    repeat_pen * self.repeat_penalty_coef + \
                    gp_reward * self.gp_reward_coef
        
        # 引入coma算法，对奖励进行信用分配
        # 先计算

        return total_reward, conc_reward, gp_reward, bound_penalty, repeat_pen, coor_pen


    # utility: expose train positions (unique)
    def train_pos(self):
        """
        构建用于 GP 训练的数据：
        - 全局：聚合所有 agent 的 (x, y, z) 观测，对重复位置进行去重（同一位置多次测量取平均）。
        - 分 agent：为每个 agent 单独构建去重后的 X/y。

        Returns
        -------
        payload : dict
            {
              'X': np.ndarray (N,2) 全局唯一位置 [x, y] in meters,
              'y': np.ndarray (N,)   全局该位置的平均测量,
              'per_agent': [
                   {'agent': i, 'X': np.ndarray (Ni,2), 'y': np.ndarray (Ni,)}, ...
              ],
              'env': {'Lx','Ly','dx','dy','grid_size'}
            }
        """
        # 构建每个 agent 的原始样本 (x,y,z)
        per_agent_samples = []
        for i in range(self.num_agents):
            traj = self.trajectories[i]
            meas = self.meas_values[i]
            n = min(len(traj), len(meas))
            if n == 0:
                per_agent_samples.append(np.zeros((0, 3), dtype=np.float32))
                continue
            xy = np.array(traj[:n], dtype=np.float32)  # (n,2)
            z = np.array(meas[:n], dtype=np.float32).reshape(-1, 1)  # (n,1)
            per_agent_samples.append(np.concatenate([xy, z], axis=1))  # (n,3)

        # 分 agent 去重（相同 (x,y) 的 z 求平均）
        per_agent_payload = []
        try:
            for i, arr in enumerate(per_agent_samples):
                if arr.shape[0] == 0:
                    per_agent_payload.append({'agent': i, 'X': np.zeros((0, 2), dtype=np.float32), 'y': np.zeros((0,), dtype=np.float32)})
                    continue
                df = pd.DataFrame(arr, columns=['x', 'y', 'z'])
                g = df.groupby(['x', 'y'], as_index=False)['z'].mean()
                X_i = g[['x', 'y']].to_numpy(dtype=np.float32)
                y_i = g['z'].to_numpy(dtype=np.float32)
                per_agent_payload.append({'agent': i, 'X': X_i, 'y': y_i})
        except Exception:
            for i, arr in enumerate(per_agent_samples):
                if arr.shape[0] == 0:
                    per_agent_payload.append({'agent': i, 'X': np.zeros((0, 2), dtype=np.float32), 'y': np.zeros((0,), dtype=np.float32)})
                    continue
                xy = arr[:, :2]
                z = arr[:, 2]
                xy_unique, inv = np.unique(xy, axis=0, return_inverse=True)
                y_accum = np.zeros((xy_unique.shape[0],), dtype=np.float64)
                counts = np.zeros((xy_unique.shape[0],), dtype=np.int64)
                for idx, val in zip(inv, z):
                    y_accum[idx] += float(val)
                    counts[idx] += 1
                y_mean = (y_accum / np.maximum(counts, 1)).astype(np.float32)
                per_agent_payload.append({'agent': i, 'X': xy_unique.astype(np.float32), 'y': y_mean})

        # 全局聚合：合并各 agent 样本并对 (x,y) 去重平均
        all_xyz = np.concatenate(per_agent_samples, axis=0) if len(per_agent_samples) > 0 else np.zeros((0, 3), dtype=np.float32)
        if all_xyz.shape[0] == 0:
            X = np.zeros((0, 2), dtype=np.float32)
            y = np.zeros((0,), dtype=np.float32)
        else:
            try:
                df_all = pd.DataFrame(all_xyz, columns=['x', 'y', 'z'])
                g_all = df_all.groupby(['x', 'y'], as_index=False)['z'].mean()
                X = g_all[['x', 'y']].to_numpy(dtype=np.float32)
                y = g_all['z'].to_numpy(dtype=np.float32)
            except Exception:
                xy = all_xyz[:, :2]
                z = all_xyz[:, 2]
                xy_unique, inv = np.unique(xy, axis=0, return_inverse=True)
                y_accum = np.zeros((xy_unique.shape[0],), dtype=np.float64)
                counts = np.zeros((xy_unique.shape[0],), dtype=np.int64)
                for idx, val in zip(inv, z):
                    y_accum[idx] += float(val)
                    counts[idx] += 1
                y_mean = (y_accum / np.maximum(counts, 1)).astype(np.float32)
                X = xy_unique.astype(np.float32)
                y = y_mean

        payload = {
            'X': X,
            'y': y,
            'per_agent': per_agent_payload,
            'env': {
                'Lx': float(self.Lx), 'Ly': float(self.Ly),
                'dx': float(self.dx), 'dy': float(self.dy),
                'grid_size': tuple(self.grid_size),
            }
        }
        return payload

    def render(self, mode='human', block=False, save_path: str = None):
        """
        可视化多智能体搜索过程。
        参数：
            block: 若为 True，窗口将在最后一步保持打开直到手动关闭。
            save_path: 若指定路径，则保存当前帧为静态图像。
        """

        if mode != 'human':
            return

        # ❶ 检查 figure 是否还活着，关掉后需要重新建
        if (self.fig is None) or (not plt.fignum_exists(self.fig.number)):
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.cbar = None
        else:
            self.ax.clear()
        # 绘制场
        im = self.ax.imshow(
            self.field,
            origin='lower',
            extent=[0, self.Lx, 0, self.Ly],
            cmap='plasma',
            alpha=0.8
        )

        # colorbar：只创建一次
        if self.cbar is None:
            self.cbar = self.fig.colorbar(im, ax=self.ax, label='Concentration')
        else:
            self.cbar.update_normal(im)

        # 绘制多智能体轨迹与位置
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))
        handles = []  # 用于 legend

        for i in range(self.num_agents):
            color = colors[i]
            label = f'Agent {i + 1}'

            # 绘制轨迹（线条更粗、更明显）
            if len(self.trajectories[i]) > 1:
                traj = np.array(self.trajectories[i])
                line, = self.ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    color=color,
                    linestyle='--',
                    alpha=0.9,
                    linewidth=2.5,
                    label=label
                )
                handles.append(line)
            else:
                # 没轨迹时也占位图例
                line, = self.ax.plot([], [], color=color, linestyle='--', alpha=0.9, linewidth=2.5, label=label)
                handles.append(line)
            # 绘制起点位置(空心圆)
            sx, sy = self.trajectories[i][0]
            self.ax.plot(sx, sy, 'o', markerfacecolor='none', markeredgecolor=color, markersize=10, markeredgewidth=2)
            # 绘制当前位置（实心圆）
            px, py = self.trajectories[i][-1]
            self.ax.plot(px, py, 'o', color=color, markersize=10, markeredgecolor='k')

            # 绘制当前浓度文本
            ix = min(int(py / self.dy), self.field.shape[0]-1)
            jx = min(int(px / self.dx), self.field.shape[1]-1)
            conc = self.field[ix, jx]
            self.ax.text(px + 1, py + 1, f'{conc:.2f}',
                        color='white', fontsize=8,
                        ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.5, pad=1))

        # 添加图例（放在右上角，不遮挡内容）
        self.ax.legend(handles=handles, loc='upper right', fontsize=9, frameon=True, framealpha=0.7)

        # 轴与标题
        self.ax.set_title(f'Step: {self.step_count} | Agents: {self.num_agents}')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_xlim(0, self.Lx)
        self.ax.set_ylim(0, self.Ly)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

        # 保存静态图片
        if save_path is not None:
            self.fig.savefig(save_path, dpi=300)
            print(f"Saved visualization to {save_path}")

        # ❸ 根据 block 决定是否阻塞
        self.fig.canvas.draw()
        if block:
            # 这里会真正卡住，直到你把窗口关掉
            plt.show(block=True)
        else:
            plt.pause(0.5)  # 非阻塞，给 GUI 刷新一点时间

# 绘制high info area
    def render_high_info_area(self, mode='human', block=False, save_path: str = None):
        """
        可视化当前 GP 高价值区域（high info area）。
        使用与环境相同的坐标范围 [0, Lx] × [0, Ly]。
        """
        if mode != 'human':
            return

        # 如果当前 figure 不存在或已被关闭，则重新创建
        if (self.fig is None) or (not plt.fignum_exists(self.fig.number)):
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.cbar_hia = None
        else:
            self.ax.clear()

        # 绘制场
        im = self.ax.imshow(
            self.field,
            origin='lower',
            extent=[0, self.Lx, 0, self.Ly],
            cmap='plasma',
            alpha=0.8
        )

        # # colorbar：只创建一次
        # if self.cbar_hia is None:
        #     self.cbar_hia = self.fig.colorbar(im, ax=self.ax, label='Concentration')
        # else:
        #     self.cbar_hia.update_normal(im)

        # 绘制 high info area 点
        if len(self.high_info_area) > 0:
            hia_array = np.array(self.high_info_area)
            self.ax.scatter(
                hia_array[:, 0],
                hia_array[:, 1],
                c='cyan',
                s=30,
                edgecolors='k',
                label='High Info Area'
            )

        # 轴与标题
        self.ax.set_title(f'High Info Area (size: {len(self.high_info_area)})')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_xlim(0, self.Lx)
        self.ax.set_ylim(0, self.Ly)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

        # 保存静态图片
        if save_path is not None:
            self.fig.savefig(save_path, dpi=300)
            print(f"Saved high info area visualization to {save_path}")

        # 显示 / 刷新
        self.fig.canvas.draw()
        if block:
            plt.show(block=True)
        else:
            plt.pause(0.5)

# 绘制预测结果
    def render_gp_prediction(self, mode='human', block=False, save_path: str = None):
        """
        可视化当前 GP 预测的均值与不确定性（方差）。
        使用与环境相同的坐标范围 [0, Lx] × [0, Ly]。
        """
        if mode != 'human':
            return

        # 如果当前 figure 不存在或已被关闭，则重新创建为 1x3 子图
        if (self.fig is None) or (not plt.fignum_exists(self.fig.number)):
            self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 5))
            self.cbar_mean = None
            self.cbar_var = None
        else:
            # 如果当前 ax 不是 2 个子图（例如刚刚调用过 render），则重新创建
            # 确保不与上面的 render 逻辑冲突
            try:
                if not isinstance(self.ax, (list, np.ndarray)) or len(self.ax) != 3:
                    plt.close(self.fig)
                    self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 5))
                    self.cbar_mean = None
                    self.cbar_var = None
                else:
                    for a in self.ax:
                        a.clear()
            except Exception:
                plt.close(self.fig)
                self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 5))
                self.cbar_mean = None
                self.cbar_var = None

        # 预测均值与方差（在统一评估网格上）
        mean_grid, var_grid, grid = self.gp_regressor.predict_on_eval_grid()

        # -------- 子图 1：真实场 --------
        im0 = self.ax[0].imshow(
            self.field,
            origin='lower',
            extent=[0, self.Lx, 0, self.Ly],
            cmap='plasma',
            alpha=0.8
        )
        self.ax[0].set_title('True Field')
        self.ax[0].set_xlabel('X')
        self.ax[0].set_ylabel('Y')
        # 为真实场创建或更新 colorbar
        if getattr(self, 'cbar_true', None) is None:
            self.cbar_true = self.fig.colorbar(im0, ax=self.ax[0], label='Concentration')

        # -------- 子图 2：均值 --------
        im1 = self.ax[1].imshow(
            mean_grid,
            origin='lower',
            extent=[0, self.Lx, 0, self.Ly],
            cmap='plasma',
            alpha=0.8
        )
        self.ax[1].set_title('GP Predicted Mean')
        self.ax[1].set_xlabel('X')
        self.ax[1].set_ylabel('Y')

        # 为均值创建或更新 colorbar
        if getattr(self, 'cbar_mean', None) is None:
            self.cbar_mean = self.fig.colorbar(im1, ax=self.ax[1], label='Mean')
        else:
            self.cbar_mean.update_normal(im1)

        # -------- 子图 3：方差 --------
        im2 = self.ax[2].imshow(
            var_grid,
            origin='lower',
            extent=[0, self.Lx, 0, self.Ly],
            cmap='viridis',
            alpha=0.8
        )
        self.ax[2].set_title('GP Predictive Variance')
        self.ax[2].set_xlabel('X')
        self.ax[2].set_ylabel('Y')

        # 为方差创建或更新 colorbar
        if getattr(self, 'cbar_var', None) is None:
            self.cbar_var = self.fig.colorbar(im2, ax=self.ax[2], label='Variance')
        else:
            self.cbar_var.update_normal(im2)

        self.fig.tight_layout()

        # 如需要，保存当前帧
        if save_path is not None:
            self.fig.savefig(save_path, dpi=300)
            print(f"Saved GP prediction visualization to {save_path}")

        # 显示 / 刷新
        self.fig.canvas.draw()
        if block:
            plt.show(block=True)
        else:
            plt.pause(0.5)

    # 内部方法：归一化 obs 向量
    def _normalize_obs_vector(self, obs_vec: np.ndarray) -> np.ndarray:
        """
        内部方法：把单个 obs 向量按字段归一化并返回新的 numpy array (float32).
        Assumes layout per agent (same as _get_obs builds):
          [pos_x, pos_y, self_nearby_means(K), self_nearby_vars(K),
              then for each other agent j != i:
                [agent_id, rel_x, rel_y, neighbor_nearby_mean_avg, neighbor_nearby_var_avg] ]
        """
        o = obs_vec.astype(np.float32).copy()
        Lx = float(self.Lx)
        Ly = float(self.Ly)
        idx = 0

        # pos_x, pos_y
        o[idx] = o[idx] / Lx; idx += 1
        o[idx] = o[idx] / Ly; idx += 1

        # self_nearby_means(K), self_nearby_vars(K): 目前保持原尺度（通常已经在 [0,1] 左右）
        idx += 2 * int(getattr(self, 'high_info_K', 8))

        # neighbor messages: each chunk = 5
        # (agent_id, rel_x, rel_y, neighbor_mean_avg, neighbor_var_avg)
        chunk = 5
        # compute number of neighbors expected (num_agents-1)
        n_neighbors = self.num_agents - 1
        # safety: ensure enough length
        for n in range(n_neighbors):
            # agent_id
            if idx >= o.shape[0]:
                break
            agent_id = o[idx]
            if self.num_agents > 1:
                o[idx] = agent_id / float(max(1, self.num_agents - 1))
            else:
                o[idx] = 0.0
            idx += 1
            # rel_x
            o[idx] = o[idx] / Lx; idx += 1
            # rel_y
            o[idx] = o[idx] / Ly; idx += 1

            # neighbor_mean_avg, neighbor_var_avg: 保持原尺度
            idx += 2

        return o.astype(np.float32)