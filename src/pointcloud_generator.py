"""
合成鞋底点云生成模块

用数学曲面生成鞋底形状的点云，不涉及真实数据（NDA安全）。
鞋底形状近似为一个椭圆底面 + 略微上翘的鞋头 + 弧形边缘。

生成的点云用于展示：
  点云 → 多平面投影 → 轮廓提取 → 轨迹生成 的完整链路
"""
import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ShoeConfig:
    """鞋底形状参数"""
    length: float = 2.0       # 鞋底长度
    width: float = 0.7        # 鞋底宽度
    toe_lift: float = 0.15    # 鞋头上翘高度
    heel_height: float = 0.05 # 鞋跟高度
    edge_curvature: float = 0.3  # 边缘曲率
    density: int = 2000       # 点云密度


def generate_shoe_pointcloud(config: ShoeConfig = ShoeConfig()) -> np.ndarray:
    """生成鞋底点云

    鞋底形状建模：
    - 俯视图：椭圆（长轴=长度，短轴=宽度）
    - 侧面：鞋头上翘 + 鞋跟微抬
    - 边缘：弧形过渡

    Returns:
        N×3 点云数组 (x=长度方向, y=宽度方向, z=高度)
    """
    N = config.density
    L, W = config.length, config.width

    # 在椭圆区域内随机采样
    u = np.random.uniform(0, 2 * np.pi, N)
    r = np.sqrt(np.random.uniform(0, 1, N))

    x = r * L/2 * np.cos(u)
    y = r * W/2 * np.sin(u)

    # 鞋头上翘：x > 0 的部分 z 随 x 增加
    # 鞋跟微抬：x < -L/4 的部分 z 微增
    z = np.zeros(N)

    toe_mask = x > 0
    z[toe_mask] = config.toe_lift * (x[toe_mask] / (L/2)) ** 2

    heel_mask = x < -L * 0.3
    z[heel_mask] = config.heel_height * (1 - (x[heel_mask] + L*0.3) / (L*0.2)) ** 2

    # 边缘弧形：靠近椭圆边缘的点 z 略微升高
    edge_dist = np.sqrt((x / (L/2))**2 + (y / (W/2))**2)
    edge_mask = edge_dist > 0.7
    z[edge_mask] += config.edge_curvature * (edge_dist[edge_mask] - 0.7) ** 2

    # 添加少量噪声（模拟深度相机噪声）
    noise = np.random.normal(0, 0.005, (N, 1))
    z = z + noise[:, 0]

    points = np.stack([x, y, z], axis=1)
    return points.astype(np.float32)


def generate_shoe_edge_pointcloud(config: ShoeConfig = ShoeConfig()) -> np.ndarray:
    """生成鞋底边缘点云（涂胶区域轮廓）

    涂胶区域在鞋底边缘，是一个闭合环。
    这个函数生成边缘的点云，用于轨迹提取。

    Returns:
        N×3 边缘点云
    """
    N = config.density
    L, W = config.length, config.width

    # 椭圆边缘采样
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = L/2 * np.cos(t)
    y = W/2 * np.sin(t)

    # 边缘高度
    z = np.zeros(N)

    toe_mask = x > 0
    z[toe_mask] = config.toe_lift * (x[toe_mask] / (L/2)) ** 2

    heel_mask = x < -L * 0.3
    z[heel_mask] = config.heel_height * (1 - (x[heel_mask] + L*0.3) / (L*0.2)) ** 2

    # 边缘本身有弧度
    z += config.edge_curvature * 0.3

    # 噪声
    noise = np.random.normal(0, 0.003, (N, 1))
    z = z + noise[:, 0]

    points = np.stack([x, y, z], axis=1)
    return points.astype(np.float32)


def generate_sole_shape(name: str = "sneaker") -> Tuple[np.ndarray, np.ndarray]:
    """生成不同鞋型的点云

    Args:
        name: 鞋型名称
            - "sneaker": 运动鞋（长+宽+鞋头上翘）
            - "leather": 皮鞋（较短+窄+微翘）
            - "boot": 靴子（长+窄+高跟）
            - "flat": 平底鞋（短+宽+无翘）

    Returns:
        (完整点云, 边缘点云)
    """
    configs = {
        "sneaker": ShoeConfig(length=2.0, width=0.7, toe_lift=0.15, heel_height=0.05, density=2000),
        "leather": ShoeConfig(length=1.8, width=0.6, toe_lift=0.08, heel_height=0.04, density=2000),
        "boot":    ShoeConfig(length=2.2, width=0.65, toe_lift=0.10, heel_height=0.12, density=2000),
        "flat":    ShoeConfig(length=1.6, width=0.75, toe_lift=0.03, heel_height=0.02, density=2000),
    }

    config = configs.get(name, configs["sneaker"])
    full_cloud = generate_shoe_pointcloud(config)
    edge_cloud = generate_shoe_edge_pointcloud(config)

    return full_cloud, edge_cloud


SHOE_TYPES = ["sneaker", "leather", "boot", "flat"]
