"""
3D 可视化模块 — 点云 / 轮廓 / 轨迹 / 法向量 / 切面

输出6种可视化：
1. 原始点云（鞋底3D形状）
2. 多平面投影切面（展示切面如何切割点云）
3. 提取的轮廓（融合后的闭合轮廓）
4. 最终轨迹（带法向量的轨迹点）
5. 全流程对比（点云→轮廓→轨迹 并排）
6. 曲率热力图（轮廓上每个点的曲率）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Dict, Optional, Tuple


def visualize_pointcloud(
    points: np.ndarray,
    title: str = "Point Cloud",
    color: str = "steelblue",
    size: float = 1.5,
    figsize: Tuple[int, int] = (8, 8),
) -> str:
    """可视化点云"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=size, alpha=0.6)

    ax.set_xlabel("X (Length)")
    ax.set_ylabel("Y (Width)")
    ax.set_zlabel("Z (Height)")
    ax.set_title(title, fontsize=14, fontweight="bold")

    max_range = max(
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min(),
    ) / 2
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    path = "output_pointcloud.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def visualize_slices(
    points: np.ndarray,
    slices: List[np.ndarray],
    planes: List,
    title: str = "Multi-Plane Projection Slices",
) -> str:
    """可视化多平面投影切面"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c="lightgray", s=0.5, alpha=0.2, label="Point Cloud")

    colors = plt.cm.Set2(np.linspace(0, 1, len(slices)))
    for i, (slice_pts, plane) in enumerate(zip(slices, planes)):
        if len(slice_pts) == 0:
            continue
        ax.scatter(slice_pts[:, 0], slice_pts[:, 1], slice_pts[:, 2],
                   c=[colors[i]], s=2, alpha=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    path = "output_slices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def visualize_contour(
    points: np.ndarray,
    contour: np.ndarray,
    title: str = "Extracted Contour",
) -> str:
    """可视化提取的轮廓（点云+轮廓叠加）"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c="lightgray", s=1, alpha=0.3, label="Point Cloud")

    ax.scatter(contour[:, 0], contour[:, 1], contour[:, 2],
               c="red", s=5, alpha=0.9, label="Contour")

    # 连接轮廓点
    if len(contour) > 2:
        contour_closed = np.vstack([contour, contour[0]])
        ax.plot(contour_closed[:, 0], contour_closed[:, 1], contour_closed[:, 2],
                c="red", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    path = "output_contour.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def visualize_trajectory(
    contour: np.ndarray,
    trajectory: np.ndarray,
    normals: np.ndarray,
    title: str = "Gluing Trajectory with Normals",
    show_normals: bool = True,
    normal_scale: float = 0.1,
) -> str:
    """可视化带法向量的涂胶轨迹"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 轮廓（灰色细线）
    if len(contour) > 2:
        contour_closed = np.vstack([contour, contour[0]])
        ax.plot(contour_closed[:, 0], contour_closed[:, 1], contour_closed[:, 2],
                c="lightgray", linewidth=1, alpha=0.4)

    # 轨迹点（彩色，按顺序）
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
    ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
               c=colors, s=20, alpha=0.9)

    # 连接轨迹点
    if len(trajectory) > 2:
        traj_closed = np.vstack([trajectory, trajectory[0]])
        ax.plot(traj_closed[:, 0], traj_closed[:, 1], traj_closed[:, 2],
                c="blue", linewidth=2, alpha=0.7)

    # 法向量箭头
    if show_normals and len(trajectory) > 0:
        step = max(1, len(trajectory) // 20)
        for i in range(0, len(trajectory), step):
            p = trajectory[i]
            n = normals[i]
            ax.quiver(p[0], p[1], p[2],
                      n[0]*normal_scale, n[1]*normal_scale, n[2]*normal_scale,
                      color="green", alpha=0.6, linewidth=1.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    path = "output_trajectory.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def visualize_full_pipeline(
    points: np.ndarray,
    contour: np.ndarray,
    trajectory: np.ndarray,
    normals: np.ndarray,
) -> str:
    """全流程对比可视化（3个子图并排）"""
    fig = plt.figure(figsize=(18, 6))

    # 1. 原始点云
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                c="steelblue", s=1, alpha=0.5)
    ax1.set_title("1. Point Cloud\n(Shoe Sole)", fontsize=12, fontweight="bold")
    ax1.view_init(elev=30, azim=45)

    # 2. 提取的轮廓
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                c="lightgray", s=0.5, alpha=0.2)
    if len(contour) > 2:
        contour_closed = np.vstack([contour, contour[0]])
        ax2.plot(contour_closed[:, 0], contour_closed[:, 1], contour_closed[:, 2],
                 c="red", linewidth=2)
    ax2.scatter(contour[:, 0], contour[:, 1], contour[:, 2],
                c="red", s=5)
    ax2.set_title("2. Contour\n(Multi-Plane Projection)", fontsize=12, fontweight="bold")
    ax2.view_init(elev=30, azim=45)

    # 3. 最终轨迹
    ax3 = fig.add_subplot(133, projection="3d")
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
    ax3.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                c=colors, s=20)
    if len(trajectory) > 2:
        traj_closed = np.vstack([trajectory, trajectory[0]])
        ax3.plot(traj_closed[:, 0], traj_closed[:, 1], traj_closed[:, 2],
                 c="blue", linewidth=2)
    # 法向量
    step = max(1, len(trajectory) // 10)
    for i in range(0, len(trajectory), step):
        p = trajectory[i]
        n = normals[i]
        ax3.quiver(p[0], p[1], p[2],
                   n[0]*0.1, n[1]*0.1, n[2]*0.1,
                   color="green", alpha=0.6, linewidth=1)
    ax3.set_title("3. Trajectory\n(With Normals)", fontsize=12, fontweight="bold")
    ax3.view_init(elev=30, azim=45)

    plt.tight_layout()
    path = "output_pipeline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def visualize_curvature(
    contour: np.ndarray,
    curvatures: np.ndarray,
    title: str = "Contour Curvature Heatmap",
) -> str:
    """曲率热力图"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 归一化曲率为颜色
    curv_norm = curvatures / (curvatures.max() + 1e-8)
    colors = plt.cm.hot(curv_norm)

    ax.scatter(contour[:, 0], contour[:, 1], contour[:, 2],
               c=colors, s=10, alpha=0.9)

    if len(contour) > 2:
        contour_closed = np.vstack([contour, contour[0]])
        ax.plot(contour_closed[:, 0], contour_closed[:, 1], contour_closed[:, 2],
                c="gray", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vmin=0, vmax=curvatures.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label="Curvature (1/m)")

    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    path = "output_curvature.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
