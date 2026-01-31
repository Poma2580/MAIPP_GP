"""
GP_utils: 统一的高斯过程回归接口

此模块提供了一种简洁的方式来使用 GPytorch 执行高斯过程回归（Gaussian Process Regression, GPR），支持全局数据和按 agent 数据选择进行训练和预测。模型内部封装了 GP 回归的各项操作，如训练数据的选择、评估网格的生成、预测接口等，使得高斯过程的使用更加简便和高效。

用法（全局数据）：
    reg = GPExactRegressor(payload, gp_params={'lengthscale': 5.0, 'outputscale': 1.0, 'noise': 1e-3}, grid_N=100)
    mean_np, var_np, grid = reg.predict_on_eval_grid()

用法（按 agent 数据）：
    reg = GPExactRegressor(payload, gp_params, grid_N=100)
    reg.select_agent_dataset(agent_id=0)
    mean_np, var_np, grid = reg.predict_on_eval_grid()

说明：
    - payload 由环境的 train_pos() 返回，包含全局位置与每个 agent 的独立数据：
        payload = {
          'X': (N,2)      # 全局位置 [x, y]
          'y': (N,)       # 对应的观测值
          'per_agent': [{'agent': i, 'X': (Ni, 2), 'y': (Ni,)}], # 每个 agent 的独立数据集
          'env': {        # 环境的配置信息
            'Lx': float,  # 地图的 X 方向长度
            'Ly': float,  # 地图的 Y 方向长度
            'dx': float,  # 网格大小，X 方向
            'dy': float,  # 网格大小，Y 方向
            'grid_size': (Ny, Nx)  # 网格的大小 (行数, 列数)
          }
        }
    - gp_params 由外部配置/参数传入，指定 GP 的超参数：
        gp_params = {
          'lengthscale': float,  # 高斯过程的长度尺度
          'outputscale': float,  # 输出尺度
          'noise': float         # 噪声项
        }

    - 支持全局数据集与单个 agent 数据集的选择，提供灵活的训练与预测接口。
    - 在初始化时，会构建均匀的评估网格，并提供在网格上进行预测的接口。
    - 可在训练过程中动态更新训练数据，而无需重建整个模型实例。

功能：
    - **全局数据训练**：使用整个环境的训练数据集进行高斯过程回归。
    - **按 agent 数据训练**：为每个 agent 单独训练，便于实现局部数据的独立建模与预测。
    - **评估网格预测**：在预定义的网格上进行高斯过程预测，输出均值和方差。
    - **训练数据更新**：支持在预测时动态更新训练数据，减少模型重建的开销。

返回值：
    - **mean_np**: 预测的均值，形状为 (N*N,)
    - **var_np**: 预测的方差，形状为 (N*N,)
    - **grid**: 评估网格的坐标信息，包括 X 和 Y 坐标，形状为 (N, N)
"""

import numpy as np
import torch
import gpytorch
from typing import Dict, Optional, Tuple
from scipy.spatial import distance
from dataclasses import dataclass
from typing import Union  # 添加这个导入
@dataclass
class GPParams:
    """GP参数的简单封装"""
    lengthscale: float
    outputscale: float
    noise: float


class _ExactGPModel(gpytorch.models.ExactGP):
    """GPytorch的高斯过程模型，使用RBF核"""
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPExactRegressor:
    """基于 GPyTorch 的 Exact GP 封装，支持全局或按 agent 的数据，统一预测接口。"""

    def __init__(self, payload: Dict, gp_params: Union[Dict, GPParams], grid_N: int = 100, device: str = "cuda", dtype: torch.dtype = torch.float32) -> None:
        self.payload = payload
        self.env = payload.get('env', {})
        self.grid_N = int(grid_N)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # 解析 GP 参数
        if isinstance(gp_params, dict):
            self.gp_params = GPParams(
                lengthscale=float(gp_params['lengthscale']),
                outputscale=float(gp_params['outputscale']),
                noise=float(gp_params['noise']),
            )
        else:
            self.gp_params = gp_params

        # 默认数据集：全局聚合数据
        self.set_dataset_global()

        # 初始化评估网格（N x N 均匀划分）
        self._build_eval_grid()

        # 初始化模型
        self._build_model()

    # ----------------------------- 数据集选择 -----------------------------
    def set_dataset(self, X: np.ndarray, y: np.ndarray) -> None:
        """设置当前训练数据集。X: (N,2), y: (N,)"""
        self.X_np = np.asarray(X, dtype=np.float32)
        self.y_np = np.asarray(y, dtype=np.float32).reshape(-1)
        # 构建 torch tensor
        self.train_x = torch.tensor(self.X_np, dtype=self.dtype)
        self.train_y = torch.tensor(self.y_np, dtype=self.dtype)

    def set_dataset_global(self) -> None:
        """从 payload 中提取全局数据并设置数据集"""
        X = self.payload.get('X', np.zeros((0, 2), dtype=np.float32))
        y = self.payload.get('y', np.zeros((0,), dtype=np.float32))
        self.set_dataset(X, y)

    def select_agent_dataset(self, agent_id: int) -> None:
        """选择特定 agent 的数据集"""
        per_agent = self.payload.get('per_agent', [])
        matches = [d for d in per_agent if int(d.get('agent', -1)) == int(agent_id)]
        if not matches:
            raise ValueError(f"未找到 agent_id={agent_id} 的数据。")
        X_i = matches[0].get('X', np.zeros((0, 2), dtype=np.float32))
        y_i = matches[0].get('y', np.zeros((0,), dtype=np.float32))
        self.set_dataset(X_i, y_i)
        self._build_model()  # 重新构建模型以适应新数据集

    def update_dataset_from_payload(self, payload: Dict, agent_id: Optional[int] = None) -> None:
        """从外部的 payload 数据更新训练数据"""
        if agent_id is None:
            # 使用全局数据集
            X = payload.get('X', np.zeros((0, 2), dtype=np.float32))
            y = payload.get('y', np.zeros((0,), dtype=np.float32))
            self.set_dataset(X, y)
        else:
            # 使用特定 agent 的数据集
            per_agent = payload.get('per_agent', [])
            matches = [d for d in per_agent if int(d.get('agent', -1)) == int(agent_id)]
            if not matches:
                raise ValueError(f"未找到 agent_id={agent_id} 的数据。")
            X_i = matches[0].get('X', np.zeros((0, 2), dtype=np.float32))
            y_i = matches[0].get('y', np.zeros((0,), dtype=np.float32))
            self.set_dataset(X_i, y_i)
        
        self._build_model()  # 更新模型

    # ----------------------------- 网格与模型 -----------------------------
    def _build_eval_grid(self) -> None:
        Lx = float(self.env.get('Lx', 1.0))
        Ly = float(self.env.get('Ly', 1.0))
        gx = np.linspace(0.0, Lx, self.grid_N, dtype=np.float32)
        gy = np.linspace(0.0, Ly, self.grid_N, dtype=np.float32)
        Xg, Yg = np.meshgrid(gx, gy, indexing='xy')  # shape (N,N)
        test_xy = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)  # (N*N,2)
        self.eval_grid = {
            'Xg': Xg,  # (N,N)
            'Yg': Yg,  # (N,N)
            'test_xy_np': test_xy,  # (N*N,2)
        }
        self.test_x = torch.tensor(test_xy, dtype=self.dtype).to(self.device)

    def _build_model(self) -> None:
        """构建或重建模型"""
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if self.train_x.shape[0] == 0:
            self.train_x = torch.zeros((1, 2), dtype=self.dtype, device=self.device)
            self.train_y = torch.zeros((1,), dtype=self.dtype, device=self.device)
        self.model = _ExactGPModel(self.train_x, self.train_y, self.likelihood)

        # 设置超参数并冻结
        self.model.covar_module.base_kernel.lengthscale = torch.tensor(self.gp_params.lengthscale, dtype=self.dtype, device=self.device)
        self.model.covar_module.outputscale = torch.tensor(self.gp_params.outputscale, dtype=self.dtype, device=self.device)
        self.likelihood.noise = torch.tensor(self.gp_params.noise, dtype=self.dtype, device=self.device)

        self.model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
        self.model.covar_module.raw_outputscale.requires_grad = False
        self.likelihood.raw_noise.requires_grad = False
        self.model.mean_module.constant.requires_grad = False
         
        # 把整个model 和 likelihood 移动到指定设备
        self.model.to(self.device, dtype=self.dtype)
        self.likelihood.to(self.device, dtype=self.dtype)

        self.model.eval()
        self.likelihood.eval()

    # ----------------------------- 预测接口 -----------------------------
    def predict(self, X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """对给定测试点做 GP 预测"""
        if X_test is None:
            test_x = self.test_x
        else:
            test_x = torch.tensor(np.asarray(X_test, dtype=np.float32), dtype=self.dtype).to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model(test_x)
            pred_mean = posterior.mean
            pred_var = posterior.variance
        
        return pred_mean.detach().cpu().numpy(), pred_var.detach().cpu().numpy()

    def predict_on_eval_grid(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """在均匀评估网格上进行预测"""
        mean_np, var_np = self.predict()
        N = self.grid_N
        mean_grid = mean_np.reshape(N, N)
        var_grid = var_np.reshape(N, N)
        return mean_grid, var_grid, {'Xg': self.eval_grid['Xg'], 'Yg': self.eval_grid['Yg']}

    # ----------------------------- 返回局部协方差 -----------------------------
    def get_local_covariance(self, agent_id: int, agent_pos: np.ndarray, local_X: np.ndarray, local_var: np.ndarray, num_neighbors: int = 8) -> np.ndarray:
        """获取指定 agent 的局部协方差（以周围最近的 num_neighbors 个点为依据）（方形）"""
        
        # 1. 获取当前 agent 的位置
        x_agent, y_agent = agent_pos

        # # 2. 获取评估网格的所有点（网格大小）
        # Xg = self.eval_grid['Xg']
        # Yg = self.eval_grid['Yg']
        
        # 3. 计算评估网格每个点与 agent 位置的距离
        grid_points = local_X  # 使用传入的局部点坐标
        agent_pos_2d = np.array([x_agent, y_agent])
        
        dist = distance.cdist([agent_pos_2d], grid_points)[0]  # 距离向量
        
        # 4. 找到距离最近的 num_neighbors 个网格点
        nearest_idx = np.argsort(dist)[:num_neighbors]
        
        # 5. 获取对应点的协方差值
        local_covariance = local_var.ravel()[nearest_idx]  # 使用 ravel 将 grid flatten 为一维数组
        
        return local_covariance
    
    ## 用圆形邻域获取局部协方差
    def get_local_covariance_circular(
        self,
        agent_id: int,
        agent_pos: np.ndarray,
        radius: float = 4.0,
    ):
        """获取指定 agent 的局部协方差（以半径 radius 内的点为依据）（圆形）
        只对圆形邻域内的网格点做 GP 预测，返回这些点的坐标和方差。
        """
        # 1. 获取当前 agent 的位置
        x_agent, y_agent = agent_pos

        # 2. 获取评估网格的所有点（网格大小）
        Xg = self.eval_grid['Xg']  # (N, N)
        Yg = self.eval_grid['Yg']  # (N, N)

        # 3. 将网格点展开为 (N*N, 2)
        grid_points = np.vstack([Xg.ravel(), Yg.ravel()]).T.astype(np.float32)
        agent_pos_2d = np.array([x_agent, y_agent], dtype=np.float32)

        # 4. 计算每个网格点与 agent 位置的距离
        dist = distance.cdist([agent_pos_2d], grid_points)[0]

        # 5. 找到距离在 radius 范围内的网格点索引
        within_radius_idx = np.where(dist <= radius)[0]

        if within_radius_idx.size == 0:
            # 半径内没有点时返回空
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # 6. 取出圆形邻域内的局部网格点坐标
        X_test_local = grid_points[within_radius_idx]  # (K, 2)

        # 7. 只对这些局部点做 GP 预测，得到局部方差
        _, local_var = self.predict(X_test=X_test_local)  # local_var: (K,)

        return X_test_local, local_var
# ----------------------------- 计算全局RMSE -----------------------------
    def compute_rmse(self, true_field, mean_grid: np.ndarray) -> float:
        """计算在评估网格上的 RMSE，true_field 形状应为 (N, N)"""
        rmse = np.sqrt(np.mean((mean_grid - true_field) ** 2))
        return rmse