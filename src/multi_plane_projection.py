"""
多平面投影模块 — 从点云提取截面轮廓

核心算法（和专利方法一致）：
1. 从不同方向对点云做切面（多平面投影）
2. 每个截面获取轮廓点（最高点/边缘点）
3. 融合多个方向的截面特征
4. 拟合连续、平滑、闭合的三维轮廓

为什么用多平面投影而非单方向扫描：
鞋底是复杂曲面，单方向扫描可能在曲率变化剧烈的区域丢失轮廓点。
多平面投影从多个方向获取截面，融合后覆盖鞋底完整轮廓。
"""
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class Plane:
    """切平面定义"""
    normal: np.ndarray   # 平面法向量
    offset: float        # 平面偏移量
    direction: str       # 投影方向描述


def generate_projection_planes(num_planes: int = 8) -> List[Plane]:
    """生成多方向投影平面

    在 XY 平面上均匀分布 N 个方向的切面。
    每个切面垂直于 XY 平面，沿不同角度切割点云。

    Args:
        num_planes: 切面数量（越多越精细，但越慢）

    Returns:
        平面列表
    """
    planes = []
    angles = np.linspace(0, np.pi, num_planes, endpoint=False)

    for i, angle in enumerate(angles):
        normal = np.array([np.cos(angle), np.sin(angle), 0])
        planes.append(Plane(
            normal=normal,
            offset=0,
            direction=f"plane_{i}_{angle:.1f}deg",
        ))

    return planes


def project_to_plane(
    points: np.ndarray,
    plane: Plane,
    thickness: float = 0.05,
) -> np.ndarray:
    """将点云投影到某个切面

    取距离切面 ±thickness 范围内的点，投影到切面上。

    Args:
        points: N×3 点云
        plane: 切平面
        thickness: 切面厚度

    Returns:
        M×3 投影后的点
    """
    distances = points @ plane.normal + plane.offset
    mask = np.abs(distances) < thickness
    sliced = points[mask].copy()

    if len(sliced) == 0:
        return np.empty((0, 3))

    projected = sliced - np.outer(distances[mask], plane.normal)
    return projected


def extract_contour_from_slice(
    slice_points: np.ndarray,
    plane: Plane,
) -> np.ndarray:
    """从截面点中提取轮廓点

    将截面点投影到切面的2D坐标系，找到上边缘点（最高点）。
    在涂胶场景中，涂胶区域在鞋底边缘的最高处。

    Args:
        slice_points: 截面点云
        plane: 切平面

    Returns:
        K×3 轮廓点
    """
    if len(slice_points) < 3:
        return slice_points

    # 构建切面的2D坐标系
    # u轴: 切面法向量在XY平面的垂直方向
    # v轴: Z轴
    u_axis = np.array([-plane.normal[1], plane.normal[0], 0])
    u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-8)
    v_axis = np.array([0, 0, 1])

    u_coords = slice_points @ u_axis
    v_coords = slice_points @ v_axis

    # 按u坐标分bin，每个bin取最高点（v最大的点）
    num_bins = min(50, len(slice_points) // 3)
    if num_bins < 3:
        return slice_points

    u_min, u_max = u_coords.min(), u_coords.max()
    if u_max - u_min < 1e-6:
        return slice_points

    bin_edges = np.linspace(u_min, u_max, num_bins + 1)
    bin_indices = np.clip(((u_coords - u_min) / (u_max - u_min) * num_bins).astype(int), 0, num_bins - 1)

    contour_indices = []
    for b in range(num_bins):
        bin_mask = bin_indices == b
        if not np.any(bin_mask):
            continue
        bin_v = v_coords[bin_mask]
        max_idx = np.where(bin_mask)[0][np.argmax(bin_v)]
        contour_indices.append(max_idx)

    if not contour_indices:
        return slice_points

    return slice_points[contour_indices]


def multi_plane_projection(
    points: np.ndarray,
    num_planes: int = 8,
    slice_thickness: float = 0.05,
) -> Dict:
    """多平面投影轮廓提取

    完整流程：
    1. 生成多方向切面
    2. 每个切面提取截面 + 轮廓点
    3. 融合所有轮廓点
    4. 按角度排序形成闭合轮廓

    Args:
        points: N×3 点云
        num_planes: 切面数量
        slice_thickness: 切面厚度

    Returns:
        {
            "contour": 融合后的轮廓点 K×3,
            "slices": 每个切面的截面点列表,
            "slice_contours": 每个切面的轮廓点列表,
            "planes": 切平面列表,
        }
    """
    print(f"  多平面投影: {num_planes} 个切面, 厚度={slice_thickness}")

    planes = generate_projection_planes(num_planes)
    all_contour_points = []
    slices = []
    slice_contours = []

    for plane in planes:
        slice_points = project_to_plane(points, plane, slice_thickness)
        slices.append(slice_points)

        contour = extract_contour_from_slice(slice_points, plane)
        slice_contours.append(contour)
        all_contour_points.append(contour)

    if all_contour_points and len(all_contour_points[0]) > 0:
        all_contour = np.vstack([c for c in all_contour_points if len(c) > 0])
    else:
        all_contour = np.empty((0, 3))

    # 按角度排序形成闭合轮廓
    if len(all_contour) > 0:
        angles = np.arctan2(all_contour[:, 1], all_contour[:, 0])
        sort_idx = np.argsort(angles)
        contour_sorted = all_contour[sort_idx]
    else:
        contour_sorted = all_contour

    print(f"  轮廓点数: {len(contour_sorted)}")

    return {
        "contour": contour_sorted,
        "slices": slices,
        "slice_contours": slice_contours,
        "planes": planes,
    }


def smooth_contour(contour: np.ndarray, window_size: int = 5) -> np.ndarray:
    """平滑轮廓（移动平均）

    多平面投影的轮廓点可能有抖动，用移动平均平滑。

    Args:
        contour: K×3 轮廓点
        window_size: 平滑窗口

    Returns:
        K×3 平滑后的轮廓
    """
    if len(contour) < window_size:
        return contour

    smoothed = contour.copy()
    for i in range(len(contour)):
        start = max(0, i - window_size // 2)
        end = min(len(contour), i + window_size // 2 + 1)
        smoothed[i] = contour[start:end].mean(axis=0)

    return smoothed
