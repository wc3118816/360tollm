"""M017 Mesh 重建: 从深度图生成三角网格.

算法: 有序网格三角化 (organized grid triangulation).
- 深度图 (H, W) 天然有序, 每个像素对应一个 3D 顶点.
- 每个 2x2 像素块生成 2 个三角形:
    [v0, v1, v2] 和 [v1, v3, v2]
    其中 v0=(r,c) v1=(r,c+1) v2=(r+1,c) v3=(r+1,c+1).
- 无效深度 (<= min_depth 或 NaN 或 > max_depth) 的顶点跳过.
- 深度跳变 (相邻像素深度差 > max_edge_length) 的三角形跳过 → 破洞.

优点:
- 实现极简 (纯 numpy 向量化).
- 保留完整像素邻接关系, 无需 kNN.
- 适合全景图 (equirect) 和透视图 (perspective).

统计:
- hole_ratio = 被跳过的三角形数 / 总可能三角形数.
- n_faces = 实际三角形数.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from modules.depth.types import DepthMap
from modules.logging import get_logger
from modules.projection import pixel_to_ray
from modules.vo.types import Pose

from .types import TriangleMesh

_LOG = get_logger("modules.mesh.generator")

# 深度无效值阈值.
_DEFAULT_MIN_DEPTH = 0.1
_DEFAULT_MAX_DEPTH = 100.0
# 深度跳变阈值 (米), 相邻像素深度差超过此值则跳过该三角形.
_DEFAULT_MAX_EDGE = 10.0
# 内参维度.
_K_SIZE = 3
# 三角面顶点数.
_FACE_VERTS = 3
# 颜色通道.
_RGB_CHANNELS = 3


@dataclass
class MeshGeneratorConfig:
    """网格生成配置."""

    projection: Literal["perspective", "equirect"] = "equirect"
    min_depth: float = _DEFAULT_MIN_DEPTH
    max_depth: float = _DEFAULT_MAX_DEPTH
    max_edge_length: float = _DEFAULT_MAX_EDGE  # 相邻顶点最大距离 (米)
    step: int = 1  # 降采样步长 (1=全像素, 2=每隔一个)


class MeshGenerator:
    """深度图 → 三角网格.

    用法:
        gen = MeshGenerator(MeshGeneratorConfig(projection="equirect"))
        mesh = gen.generate(depth_map, pose)
        mesh.save_gltf("output.gltf")
    """

    def __init__(self, cfg: MeshGeneratorConfig | None = None) -> None:
        self.cfg = cfg or MeshGeneratorConfig()
        self._log = _LOG.bind(projection=self.cfg.projection)

    def generate(  # noqa: PLR0915
        self,
        depth_map: DepthMap,
        pose: Pose,
        image: np.ndarray | None = None,
        camera_matrix: np.ndarray | None = None,
    ) -> TriangleMesh:
        """深度图 → 三角网格 (世界坐标系).

        Args:
            depth_map: DepthMap (M009).
            pose: Pose (M011), world_to_camera.
            image: RGB uint8 (H, W, 3), 可选 (顶点颜色).
            camera_matrix: 3x3 内参 K. perspective 模式必需.

        Returns:
            TriangleMesh.
        """
        H, W = depth_map.height, depth_map.width
        step = self.cfg.step

        # 1. 生成顶点 (camera 坐标系), 保留 (H, W) 网格结构.
        v_coords, u_coords = np.mgrid[0:H:step, 0:W:step]
        h_s, w_s = v_coords.shape  # 降采样后的 H, W.

        # 计算每个像素的射线.
        if self.cfg.projection == "perspective":
            if camera_matrix is None:
                raise ValueError("perspective mode requires camera_matrix K")
            K = np.asarray(camera_matrix, dtype=np.float64)
            if K.shape != (_K_SIZE, _K_SIZE):
                raise ValueError(f"camera_matrix must be (3,3), got {K.shape}")
            K_inv = np.linalg.inv(K)
            u_flat = u_coords.ravel().astype(np.float64)
            v_flat = v_coords.ravel().astype(np.float64)
            ones = np.ones_like(u_flat)
            uv1 = np.stack([u_flat, v_flat, ones], axis=0)
            rays_flat = K_inv @ uv1  # (3, N)
            rays = rays_flat.T.reshape(h_s, w_s, _FACE_VERTS)
        else:  # equirect
            u_flat = u_coords.ravel().astype(np.float64)
            v_flat = v_coords.ravel().astype(np.float64)
            rays_flat = np.array(
                [pixel_to_ray(u, v, W, H) for u, v in zip(u_flat, v_flat, strict=True)]
            )
            rays = rays_flat.reshape(h_s, w_s, _FACE_VERTS)

        # 深度图 (降采样).
        depth = depth_map.depth[::step, ::step].astype(np.float64)  # (h_s, w_s)

        # 顶点 (camera 坐标系): point_cam = depth * ray.
        points_cam = depth[:, :, None] * rays  # (h_s, w_s, 3)

        # 2. 有效深度 mask.
        valid = (
            (depth > self.cfg.min_depth) & (depth < self.cfg.max_depth) & np.isfinite(depth)
        )  # (h_s, w_s)

        # 3. 世界坐标变换.
        # point_world = (point_cam - t) @ R  (R 正交, R^T = R^{-1}).
        R = pose.R
        t = pose.t
        # (h_s, w_s, 3) → (N, 3) @ R → (h_s, w_s, 3).
        points_world = (points_cam - t) @ R  # 广播: (h_s, w_s, 3) - (3,) = (h_s, w_s, 3), @ (3,3)

        # 4. 三角化: 2x2 像素块 → 2 个三角形.
        # 顶点索引: 按行优先排列. idx(r, c) = r * w_s + c.
        # 每个 2x2 块:
        #   v0 = (r, c), v1 = (r, c+1), v2 = (r+1, c), v3 = (r+1, c+1)
        #   三角形 1: [v0, v1, v2]
        #   三角形 2: [v1, v3, v2]

        r_idx, c_idx = np.mgrid[0 : h_s - 1, 0 : w_s - 1]
        r_flat = r_idx.ravel()
        c_flat = c_idx.ravel()
        n_quads = len(r_flat)

        v0_idx = r_flat * w_s + c_flat
        v1_idx = r_flat * w_s + (c_flat + 1)
        v2_idx = (r_flat + 1) * w_s + c_flat
        v3_idx = (r_flat + 1) * w_s + (c_flat + 1)

        # 4 个顶点的有效 mask.
        v0_valid = valid[r_flat, c_flat]
        v1_valid = valid[r_flat, c_flat + 1]
        v2_valid = valid[r_flat + 1, c_flat]
        v3_valid = valid[r_flat + 1, c_flat + 1]

        # 深度跳变 mask (防止跨边界的三角形).
        v0_pos = points_world[r_flat, c_flat]
        v1_pos = points_world[r_flat, c_flat + 1]
        v2_pos = points_world[r_flat + 1, c_flat]
        v3_pos = points_world[r_flat + 1, c_flat + 1]

        # 三角形 1 [v0, v1, v2]: 检查 3 条边.
        edge01 = np.linalg.norm(v1_pos - v0_pos, axis=1)
        edge12 = np.linalg.norm(v2_pos - v1_pos, axis=1)
        edge20 = np.linalg.norm(v0_pos - v2_pos, axis=1)
        tri1_edge_ok = (
            (edge01 < self.cfg.max_edge_length)
            & (edge12 < self.cfg.max_edge_length)
            & (edge20 < self.cfg.max_edge_length)
        )
        tri1_valid = v0_valid & v1_valid & v2_valid & tri1_edge_ok

        # 三角形 2 [v1, v3, v2]: 检查 3 条边.
        edge13 = np.linalg.norm(v3_pos - v1_pos, axis=1)
        edge32 = np.linalg.norm(v2_pos - v3_pos, axis=1)
        edge21 = np.linalg.norm(v1_pos - v2_pos, axis=1)
        tri2_edge_ok = (
            (edge13 < self.cfg.max_edge_length)
            & (edge32 < self.cfg.max_edge_length)
            & (edge21 < self.cfg.max_edge_length)
        )
        tri2_valid = v1_valid & v3_valid & v2_valid & tri2_edge_ok

        # 破洞率统计.
        total_possible = n_quads * 2
        n_holes = total_possible - int(tri1_valid.sum()) - int(tri2_valid.sum())
        hole_ratio = n_holes / total_possible if total_possible > 0 else 0.0

        # 5. 提取有效三角形.
        tri1_faces = np.stack([v0_idx, v1_idx, v2_idx], axis=1)[tri1_valid]
        tri2_faces = np.stack([v1_idx, v3_idx, v2_idx], axis=1)[tri2_valid]
        faces = np.vstack([tri1_faces, tri2_faces]).astype(np.int32)

        # 6. 提取顶点 (全部 h_s * w_s 个, 含无效点用 0 填充).
        vertices = points_world.reshape(-1, _FACE_VERTS).astype(np.float32)
        # 无效深度的顶点设为 0.
        valid_flat = valid.ravel()
        vertices[~valid_flat] = 0.0

        # 7. 顶点颜色.
        vertex_colors = np.zeros((0, _RGB_CHANNELS), dtype=np.uint8)
        if image is not None:
            img = np.asarray(image)
            if img.ndim == _RGB_CHANNELS:
                # 采样对应像素颜色.
                c_u = np.clip(u_coords.ravel(), 0, img.shape[1] - 1).astype(np.int32)
                c_v = np.clip(v_coords.ravel(), 0, img.shape[0] - 1).astype(np.int32)
                vertex_colors = img[c_v, c_u].astype(np.uint8)

        self._log.info(
            "mesh generated",
            n_vertices=int(vertices.shape[0]),
            n_faces=int(faces.shape[0]),
            hole_ratio=round(hole_ratio, 4),
            projection=self.cfg.projection,
        )

        return TriangleMesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            metadata={
                "projection": self.cfg.projection,
                "source_frame_id": pose.frame_id,
                "depth_backend": depth_map.backend,
                "image_shape": [H, W],
                "step": step,
                "hole_ratio": round(hole_ratio, 4),
                "n_holes": n_holes,
                "n_total_possible_faces": total_possible,
            },
        )

    def generate_from_pointcloud(
        self,
        points: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray | None = None,
        normals: np.ndarray | None = None,
    ) -> TriangleMesh:
        """从已有顶点 + 面索引构建 mesh (不常用, 保留接口)."""
        return TriangleMesh(
            vertices=points,
            faces=faces,
            vertex_colors=colors if colors is not None else np.zeros((0, 3), dtype=np.uint8),
            vertex_normals=normals if normals is not None else np.zeros((0, 3), dtype=np.float32),
        )
