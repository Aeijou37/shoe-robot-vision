"""
Gradio 前端 — 制鞋机器人视觉系统 Demo

展示鞋底涂胶轨迹提取的完整链路：
  合成鞋底点云 → 多平面投影 → 轮廓提取 → 轨迹生成（带法向量）

运行:
  python src/app.py
"""
import sys
import numpy as np
import gradio as gr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pointcloud_generator import generate_sole_shape, SHOE_TYPES
from src.multi_plane_projection import multi_plane_projection, smooth_contour
from src.trajectory_extraction import (
    estimate_normals,
    extract_trajectory,
    compute_trajectory_stats,
    format_trajectory_stats,
)
from src.visualization import (
    visualize_pointcloud,
    visualize_slices,
    visualize_contour,
    visualize_trajectory,
    visualize_full_pipeline,
    visualize_curvature,
)


class ShoeVisionApp:
    def __init__(self):
        print("初始化制鞋机器人视觉系统 Demo...")
        self.full_cloud = None
        self.edge_cloud = None
        self.contour = None
        self.trajectory = None
        self.normals = None
        self.curvatures = None
        print("初始化完成\n")

    def process(
        self,
        shoe_type: str,
        num_planes: int,
        slice_thickness: float,
        min_points: int,
        max_points: int,
        show_normals: bool,
    ):
        """完整处理流程"""
        # 1. 生成点云
        print(f"\n=== 处理开始: {shoe_type} ===")
        self.full_cloud, self.edge_cloud = generate_sole_shape(shoe_type)
        print(f"点云生成: {self.full_cloud.shape[0]} 点 (完整), {self.edge_cloud.shape[0]} 点 (边缘)")

        # 2. 可视化原始点云
        pc_img = visualize_pointcloud(self.full_cloud, f"Shoe Sole Point Cloud ({shoe_type})")

        # 3. 多平面投影
        proj_result = multi_plane_projection(
            self.edge_cloud,
            num_planes=num_planes,
            slice_thickness=slice_thickness,
        )
        self.contour = proj_result["contour"]

        # 4. 平滑轮廓
        if len(self.contour) > 5:
            self.contour = smooth_contour(self.contour, window_size=5)

        # 5. 可视化切面
        slices_img = visualize_slices(
            self.edge_cloud,
            proj_result["slices"],
            proj_result["planes"],
        )

        # 6. 可视化轮廓
        contour_img = visualize_contour(
            self.edge_cloud,
            self.contour,
            "Extracted Gluing Contour",
        )

        # 7. 轨迹提取
        traj_result = extract_trajectory(
            self.contour,
            min_points=min_points,
            max_points=max_points,
        )
        self.trajectory = traj_result["trajectory"]
        self.normals = traj_result["normals"]
        self.curvatures = traj_result["curvatures"]

        # 8. 可视化轨迹
        traj_img = visualize_trajectory(
            self.contour,
            self.trajectory,
            self.normals,
            show_normals=show_normals,
        )

        # 9. 全流程对比
        pipeline_img = visualize_full_pipeline(
            self.full_cloud,
            self.contour,
            self.trajectory,
            self.normals,
        )

        # 10. 曲率热力图
        from src.trajectory_extraction import compute_curvature
        all_curvatures = compute_curvature(self.contour)
        curvature_img = visualize_curvature(self.contour, all_curvatures)

        # 11. 统计信息
        stats = compute_trajectory_stats(self.trajectory)
        stats_text = format_trajectory_stats(stats)

        summary = f"""=== 处理完成: {shoe_type} ===

输入:
  鞋型: {shoe_type}
  点云: {self.full_cloud.shape[0]} 点 (完整) + {self.edge_cloud.shape[0]} 点 (边缘)

多平面投影:
  切面数: {num_planes}
  切面厚度: {slice_thickness} m
  提取轮廓点: {len(self.contour)}

轨迹提取:
  最少点数: {min_points}
  最多点数: {max_points}
  最终轨迹点: {len(self.trajectory)}

{stats_text}

技术链路:
  点云 → 多平面投影 → 轮廓提取 → 平滑 → 曲率自适应离散 → 法向量估计 → 涂胶轨迹
"""

        return pc_img, slices_img, contour_img, traj_img, pipeline_img, curvature_img, summary

    def build(self):
        with gr.Blocks(title="Shoe Robot Vision Demo") as demo:
            gr.Markdown("# Shoe Sole Gluing Trajectory Extraction Demo")
            gr.Markdown(
                "Synthetic shoe sole point cloud → multi-plane projection → contour extraction → "
                "curvature-adaptive trajectory generation with surface normals.\n\n"
                "Based on the Fujian Provincial S&T Major Project (2024.9 - 2026.3). "
                "2 patents + 1 paper. Recognition >99%, precision ≤1mm."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Parameters")

                    shoe_type = gr.Dropdown(
                        SHOE_TYPES,
                        label="Shoe Type",
                        value="sneaker",
                        info="Different shoe shapes (synthetic data, NDA-safe)",
                    )
                    num_planes = gr.Slider(
                        4, 20, value=8, step=2,
                        label="Number of Projection Planes",
                        info="More planes = finer contour but slower",
                    )
                    slice_thickness = gr.Slider(
                        0.01, 0.15, value=0.05, step=0.01,
                        label="Slice Thickness (m)",
                        info="Thickness of each projection plane",
                    )
                    min_points = gr.Slider(
                        20, 100, value=50, step=10,
                        label="Min Trajectory Points",
                    )
                    max_points = gr.Slider(
                        50, 300, value=150, step=10,
                        label="Max Trajectory Points",
                    )
                    show_normals = gr.Checkbox(
                        label="Show Normal Vectors",
                        value=True,
                        info="Display surface normals (gluing gun pose)",
                    )

                    run_btn = gr.Button("Run Pipeline", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### Results")
                    pipeline_img = gr.Image(label="Full Pipeline (Point Cloud → Contour → Trajectory)")
                    traj_img = gr.Image(label="Gluing Trajectory with Normals")

            with gr.Row():
                pc_img = gr.Image(label="1. Input Point Cloud")
                slices_img = gr.Image(label="2. Multi-Plane Slices")
                contour_img = gr.Image(label="3. Extracted Contour")

            curvature_img = gr.Image(label="4. Curvature Heatmap")
            stats_output = gr.Textbox(label="Statistics", lines=18, interactive=False)

            run_btn.click(
                fn=self.process,
                inputs=[shoe_type, num_planes, slice_thickness, min_points, max_points, show_normals],
                outputs=[pc_img, slices_img, contour_img, traj_img, pipeline_img, curvature_img, stats_output],
            )

            with gr.Tab("Help"):
                gr.Markdown("""
                ### Pipeline Overview

                ```
                Synthetic Shoe Point Cloud (NDA-safe)
                    ↓
                Multi-Plane Projection (8 directions)
                    ↓
                Contour Extraction (per-slice edge points)
                    ↓
                Smoothing (moving average)
                    ↓
                Curvature-Adaptive Discretization
                    ↓
                Normal Estimation (KNN + PCA)
                    ↓
                Gluing Trajectory (position + orientation)
                ```

                ### Key Technology

                | Step | Method | Why |
                |---|---|---|
                | Multi-plane projection | 8 directional slices | Single-direction misses curved areas |
                | Contour extraction | Per-slice highest point | Gluing area is at the sole edge |
                | Curvature-adaptive | Dense at curves, sparse at straight | Precision where needed |
                | Normal estimation | KNN + PCA | Gluing gun must be perpendicular to surface |

                ### Patent & Paper

                - **Patent 1** (First inventor): CN120852582A — 3D contour extraction with GAN error correction
                - **Patent 2** (Second inventor): CN120823398B — RGB-D camouflaged segmentation
                - **Paper** (First author, under review): Dynamic routing + topological aggregation

                ### NDA Notice

                This demo uses **synthetic data** generated by mathematical models.
                No real production data or proprietary code is included.
                The algorithms (multi-plane projection, trajectory extraction) are generic
                and published in the patents.

                ### Related Projects

                - [sam-interactive-segmentation-demo](https://github.com/Aeijou37/sam-interactive-segmentation-demo) — 2D segmentation
                - [pointcloud-demo](https://github.com/Aeijou37/pointcloud-demo) — PointNet classification
                - [edge-deployment-demo](https://github.com/Aeijou37/edge-deployment-demo) — ONNX + INT8
                - [cv-algorithm-notes](https://github.com/Aeijou37/cv-algorithm-notes) — Technical notes
                """)

        return demo


if __name__ == "__main__":
    app = ShoeVisionApp()
    demo = app.build()
    demo.launch(server_name="0.0.0.0", server_port=7860)
