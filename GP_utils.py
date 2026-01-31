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

    # -----------------------------  确定高信息区域-----------------------------
    def get_high_info_area(self, threshold: float = 0.4, beta: float = 1.0) -> np.ndarray:
        """获取高信息区域的坐标点，基于预测均值和不确定性"""
        mean_np, var_np = self.predict()
        std_np = np.sqrt(var_np)
        high_info_points = []
        for i in range(mean_np.shape[0]):
            if mean_np[i] + beta * std_np[i] >= threshold:
                high_info_points.append(self.eval_grid['test_xy_np'][i])
        return np.array(high_info_points, dtype=np.float32) # shape (M,2)

    # 获取周围最近的K个高价值区域点的均值和高斯方差（若提供了 high_info_area）
    def get_nearby_high_info(self, query_points: np.ndarray, high_info_area: Optional[np.ndarray] = None, K: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """对于给定的查询点，获取其附近K个高信息区域点的均值和方差"""
        if high_info_area is None or high_info_area.shape[0] == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        mean_list = []
        var_list = []
        for qp in query_points:
            dists = distance.cdist(high_info_area, qp.reshape(1, -1), metric='euclidean').reshape(-1)
            nearest_indices = np.argsort(dists)[:K]
            nearest_points = high_info_area[nearest_indices]

            mean_np, var_np = self.predict(nearest_points)
            mean_list.append(mean_np)
            var_list.append(var_np)

        return np.concatenate(mean_list, axis=0), np.concatenate(var_list, axis=0)
    


# ----------------------------- 计算全局RMSE -----------------------------
    def compute_rmse(self, true_field: np.ndarray, mean_grid: Optional[np.ndarray] = None) -> float:
        """计算在评估网格上的 RMSE。

        参数:
            true_field: 真实场值, 形状通常为 (N, N)。
            mean_grid:  可选, 预测的均值场。如果为 None, 则使用当前 GP 在评估网格上的预测结果。
        """
        if mean_grid is None:
            # 使用 GP 自身在评估网格上的预测
            mean_np, _ = self.predict()
            mean_grid = mean_np.reshape(self.grid_N, self.grid_N)
        else:
            mean_grid = np.asarray(mean_grid, dtype=np.float32)
            # 尽量与 true_field 的形状对齐
            if mean_grid.shape != true_field.shape:
                mean_grid = mean_grid.reshape(true_field.shape)

        rmse = np.sqrt(np.mean((mean_grid - true_field) ** 2))
        return rmse
# ----------------------------- 计算覆盖迹 -----------------------------
    def compute_cov_trace(self, X_test: Optional[np.ndarray] = None) -> float:
        """计算在评估网格(默认)上的覆盖迹"""
        _, var = self.predict(X_test)
        trace = np.sum(var*var)
        return trace
# ----------------------------- 计算互信息 -----------------------------
    def compute_mutual_info(self, X_test: Optional[np.ndarray] = None) -> float:
        """计算在评估网格(默认)上的互信息"""
        if X_test is None:
            test_x = self.test_x
        else:
            test_x = torch.tensor(np.asarray(X_test, dtype=np.float32), dtype=self.dtype).to(self.device)

        n_sample = test_x.shape[0]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            cov = self.model(test_x).covariance_matrix.detach().cpu().numpy()

        mi = (1 / 2) * np.log(np.linalg.det(0.01 * cov + np.identity(n_sample)))
        return mi