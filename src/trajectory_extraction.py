"""
轨迹提取模块 — 法向量估计 + 曲率自适应离散 + 轨迹生成

从三维轮廓生成带位姿的涂胶轨迹：
1. 曲率自适应离散化：曲率大处密、曲率小处疏
2. 法向量估计：每个轨迹点的局部曲面法向量（涂胶枪姿态）
3. 轨迹点排序：形成连续路径
4. 工艺参数注入：点间距、涂胶宽度等
"""
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree


def estimate_normals(
    points: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """估计点云法向量（KNN + PCA）

    对每个点，找K个最近邻，用PCA估计局部平面的法向量。

    Args:
        points: N×3 点云
        k: 最近邻数量

    Returns:
        N×3 法向量（单位向量）
    """
    N = len(points)
    if N < k:
        k = max(3, N)

    tree = cKDTree(points)
    normals = np.zeros_like(points)

    for i in range(N):
        _, indices = tree.query(points[i], k=k)
        neighbors = points[indices]

        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid

        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        normal = eigenvectors[:, 0]
        if normal[2] < 0:
            normal = -normal
        normals[i] = normal

    return normals


def compute_curvature(
    contour: np.ndarray,
) -> np.ndarray:
    """计算轮廓上每个点的曲率

    用三点圆拟合估计曲率：
    对于连续三个点 P1, P2, P3，曲率 κ = 1/R，R是外接圆半径。

    Args:
        contour: K×3 轮廓点

    Returns:
        K 曲率值
    """
    K = len(contour)
    curvatures = np.zeros(K)

    for i in range(K):
        p1 = contour[(i - 1) % K]
        p2 = contour[i]
        p3 = contour[(i + 1) % K]

        # 三角形边长
        a = np.linalg.norm(p3 - p2)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p2 - p1)

        # 三角形面积
        s = (a + b + c) / 2
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-12))

        # 外接圆半径 R = abc / (4*area)
        R = (a * b * c) / (4 * area + 1e-12)
        curvatures[i] = 1.0 / R

    return curvatures


def curvature_adaptive_discretization(
    contour: np.ndarray,
    curvatures: np.ndarray,
    min_points: int = 50,
    max_points: int = 200,
    curvature_threshold: float = 0.5,
) -> np.ndarray:
    """曲率自适应离散化

    曲率大的地方点多（密集），曲率小的地方点少（稀疏）。
    这样保证涂胶轨迹在弯曲处精细，在直线处高效。

    Args:
        contour: K×3 轮廓点
        curvatures: K 曲率值
        min_points: 最少轨迹点数
        max_points: 最多轨迹点数
        curvature_threshold: 曲率阈值

    Returns:
        选中作为轨迹点的索引
    """
    K = len(contour)

    # 归一化曲率
    curv_norm = curvatures / (curvatures.max() + 1e-8)

    # 按曲率排序，选曲率大的点
    num_select = int(min(max_points, max(min_points, K * 0.5)))
    
    # 曲率大的优先选
    weights = curv_norm + 0.1  # 基础权重0.1，保证曲率小的地方也有点
    weights = weights / weights.sum()

    selected = np.random.choice(K, size=min(num_select, K), replace=False, p=weights)
    selected = np.sort(selected)

    return selected


def extract_trajectory(
    contour: np.ndarray,
    normals: np.ndarray = None,
    min_points: int = 50,
    max_points: int = 200,
    point_spacing: float = 0.02,
) -> Dict:
    """从轮廓提取涂胶轨迹

    完整流程：
    1. 计算曲率
    2. 曲率自适应离散化
    3. 估计法向量（涂胶枪姿态）
    4. 排序形成连续路径

    Args:
        contour: K×3 轮廓点
        normals: 预计算的法向量（可选）
        min_points: 最少轨迹点
        max_points: 最多轨迹点
        point_spacing: 轨迹点间距 (m)

    Returns:
        {
            "trajectory": 轨迹点 M×3,
            "normals": 轨迹点法向量 M×3,
            "curvatures": 轨迹点曲率 M,
            "indices": 选中的轮廓索引,
        }
    """
    print(f"  轨迹提取: {len(contour)} 轮廓点")

    # 1. 计算曲率
    curvatures = compute_curvature(contour)
    print(f"    平均曲率: {curvatures.mean():.4f}, 最大曲率: {curvatures.max():.4f}")

    # 2. 曲率自适应离散化
    selected_idx = curvature_adaptive_discretization(
        contour, curvatures, min_points, max_points
    )
    print(f"    选中轨迹点: {len(selected_idx)}")

    # 3. 估计法向量
    if normals is None:
        normals = estimate_normals(contour, k=15)

    # 4. 提取轨迹
    trajectory = contour[selected_idx]
    traj_normals = normals[selected_idx]
    traj_curvatures = curvatures[selected_idx]

    # 5. 按角度排序（确保连续路径）
    angles = np.arctan2(trajectory[:, 1], trajectory[:, 0])
    sort_idx = np.argsort(angles)
    trajectory = trajectory[sort_idx]
    traj_normals = traj_normals[sort_idx]
    traj_curvatures = traj_curvatures[sort_idx]

    print(f"    轨迹生成完成: {len(trajectory)} 个点")

    return {
        "trajectory": trajectory,
        "normals": traj_normals,
        "curvatures": traj_curvatures,
        "indices": selected_idx,
    }


def compute_trajectory_stats(trajectory: np.ndarray) -> Dict:
    """计算轨迹统计信息"""
    if len(trajectory) < 2:
        return {"length": 0, "num_points": 0, "avg_spacing": 0}

    diffs = np.diff(trajectory, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = segment_lengths.sum()

    # 闭合轨迹：加上最后一个点到第一个点的距离
    closing = np.linalg.norm(trajectory[-1] - trajectory[0])
    total_length += closing

    return {
        "length": float(total_length),
        "num_points": len(trajectory),
        "avg_spacing": float(segment_lengths.mean()) if len(segment_lengths) > 0 else 0,
        "closing_distance": float(closing),
    }


def format_trajectory_stats(stats: Dict) -> str:
    """格式化轨迹统计"""
    return f"""涂胶轨迹统计
========================================
轨迹点数:    {stats['num_points']}
轨迹总长:    {stats['length']:.4f} m
平均点间距:  {stats['avg_spacing']*1000:.2f} mm
闭合距离:    {stats['closing_distance']*1000:.2f} mm
========================================"""
