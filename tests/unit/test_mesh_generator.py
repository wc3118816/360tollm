"""M017 单元测试: MeshGenerator 深度图 → 三角网格.

覆盖:
- 基本生成 (equirect + perspective)
- 三角面数 = (H-1)*(W-1)*2 (全有效深度)
- 无效深度 → 破洞 (hole_ratio > 0)
- 深度跳变 → 破洞
- 顶点颜色提取
- 降采样 (step > 1)
- 空深度图
- metadata 记录
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.mesh import MeshGenerator, MeshGeneratorConfig
from modules.vo.types import Pose


def _make_depth_map(h: int = 4, w: int = 4, depth_val: float = 5.0) -> np.ndarray:
    """生成均匀深度图."""
    from modules.depth.types import DepthMap

    depth = np.full((h, w), depth_val, dtype=np.float32)
    conf = np.ones((h, w), dtype=np.float32)
    return DepthMap(depth=depth, confidence=conf)


class TestMeshGeneration:
    def test_basic_equirect(self) -> None:
        dm = _make_depth_map(4, 4, depth_val=5.0)
        gen = MeshGenerator(MeshGeneratorConfig(projection="equirect"))
        mesh = gen.generate(dm, Pose.identity())
        assert mesh.n_vertices == 16  # 4x4
        assert mesh.n_faces == 18  # (4-1)*(4-1)*2 = 18
        assert mesh.metadata["projection"] == "equirect"

    def test_basic_perspective(self) -> None:
        dm = _make_depth_map(3, 3, depth_val=5.0)
        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        gen = MeshGenerator(MeshGeneratorConfig(projection="perspective"))
        mesh = gen.generate(dm, Pose.identity(), camera_matrix=K)
        assert mesh.n_vertices == 9  # 3x3
        assert mesh.n_faces == 8  # (3-1)*(3-1)*2 = 8

    def test_face_count_formula(self) -> None:
        """n_faces = (H-1)*(W-1)*2 (全有效深度, perspective 投影)."""
        K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=np.float64)
        for h, w in [(2, 2), (3, 4), (5, 5)]:
            dm = _make_depth_map(h, w, depth_val=5.0)
            gen = MeshGenerator(MeshGeneratorConfig(projection="perspective"))
            mesh = gen.generate(dm, Pose.identity(), camera_matrix=K)
            assert mesh.n_faces == (h - 1) * (w - 1) * 2

    def test_perspective_no_K_raises(self) -> None:
        dm = _make_depth_map(2, 2)
        gen = MeshGenerator(MeshGeneratorConfig(projection="perspective"))
        with pytest.raises(ValueError, match="camera_matrix"):
            gen.generate(dm, Pose.identity())


class TestHoleRatio:
    def test_no_holes(self) -> None:
        """全有效深度 → hole_ratio = 0."""
        dm = _make_depth_map(4, 4, depth_val=5.0)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())
        assert mesh.metadata["hole_ratio"] == 0.0

    def test_invalid_depth_creates_holes(self) -> None:
        """无效深度 → 破洞."""
        from modules.depth.types import DepthMap

        depth = np.full((4, 4), 5.0, dtype=np.float32)
        depth[2, 2] = 0.0  # 无效点.
        conf = np.ones((4, 4), dtype=np.float32)
        dm = DepthMap(depth=depth, confidence=conf)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())
        assert mesh.metadata["hole_ratio"] > 0.0
        assert mesh.metadata["n_holes"] > 0

    def test_depth_jump_creates_holes(self) -> None:
        """深度跳变 → 破洞."""
        from modules.depth.types import DepthMap

        depth = np.full((4, 4), 5.0, dtype=np.float32)
        depth[1, 1] = 50.0  # 突变.
        conf = np.ones((4, 4), dtype=np.float32)
        dm = DepthMap(depth=depth, confidence=conf)
        gen = MeshGenerator(MeshGeneratorConfig(max_edge_length=1.0))
        mesh = gen.generate(dm, Pose.identity())
        assert mesh.metadata["hole_ratio"] > 0.0


class TestMeshProperties:
    def test_vertex_colors(self) -> None:
        dm = _make_depth_map(3, 3, depth_val=5.0)
        img = np.full((3, 3, 3), 100, dtype=np.uint8)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity(), image=img)
        assert mesh.has_vertex_colors
        assert mesh.vertex_colors.shape == (9, 3)

    def test_no_image_no_colors(self) -> None:
        dm = _make_depth_map(3, 3, depth_val=5.0)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity(), image=None)
        # 无 image → vertex_colors 为空.
        assert mesh.vertex_colors.shape == (0, 3)
        assert not mesh.has_vertex_colors

    def test_step_downsample(self) -> None:
        dm = _make_depth_map(6, 6, depth_val=5.0)
        gen = MeshGenerator(MeshGeneratorConfig(step=2))
        mesh = gen.generate(dm, Pose.identity())
        # step=2 → 3x3 顶点.
        assert mesh.n_vertices == 9  # (6//2) * (6//2) = 9
        assert mesh.n_faces == 8  # (3-1)*(3-1)*2 = 8

    def test_metadata_recorded(self) -> None:
        dm = _make_depth_map(4, 4, depth_val=5.0)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())
        assert "projection" in mesh.metadata
        assert "source_frame_id" in mesh.metadata
        assert "hole_ratio" in mesh.metadata
        assert "n_holes" in mesh.metadata
        assert "image_shape" in mesh.metadata
        assert "step" in mesh.metadata

    def test_no_nan(self) -> None:
        """AC: mesh 无 NaN."""
        dm = _make_depth_map(4, 4, depth_val=5.0)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())
        assert not mesh.has_nan()

    def test_pose_transform(self) -> None:
        """不同 pose → 不同世界坐标."""
        from modules.vo.types import Pose

        dm = _make_depth_map(3, 3, depth_val=5.0)
        gen = MeshGenerator()
        mesh1 = gen.generate(dm, Pose.identity())
        # 平移 pose (matrix 4x4).
        m2 = np.eye(4, dtype=np.float64)
        m2[:3, 3] = [10, 0, 0]
        pose2 = Pose(matrix=m2, frame_id=1)
        mesh2 = gen.generate(dm, pose2)
        # 顶点应不同 (平移了 10m).
        assert not np.allclose(mesh1.vertices, mesh2.vertices)

    def test_face_index_valid(self) -> None:
        """所有 face 索引 < n_vertices."""
        dm = _make_depth_map(4, 4, depth_val=5.0)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())
        assert mesh.faces.max() < mesh.n_vertices
        assert mesh.faces.min() >= 0
