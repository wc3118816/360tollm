"""M017 集成测试: Image → Depth → Mesh → glTF/PLY/OBJ 端到端.

验证:
1. 完整流程: Image → Depth → Mesh
2. AC: mesh 可导出 glTF/PLY/OBJ
3. AC: 破洞率可统计
4. AC: 三角面数可统计
5. PLY 保存/加载往返
6. 完整工作流 (多格式导出)
"""

from __future__ import annotations

import json
import os

import numpy as np

from modules.depth import DummyDepthEstimator
from modules.mesh import MeshGenerator, MeshGeneratorConfig, TriangleMesh
from modules.vo import DummyVisualOdometry
from modules.vo.types import Pose


class TestFullPipeline:
    """Image → Depth → Mesh 完整链路."""

    def test_image_to_mesh(self) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())
        assert mesh.n_vertices > 0
        assert mesh.n_faces > 0

    def test_ac_no_nan(self) -> None:
        """AC: mesh 无 NaN."""
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())
        assert not mesh.has_nan()


class TestExportFormats:
    """AC: mesh 可导出 glTF/PLY/OBJ."""

    def test_export_gltf(self, tmp_path) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())

        path = str(tmp_path / "output.gltf")
        mesh.save_gltf(path)
        assert os.path.exists(path)
        bin_path = str(tmp_path / "output.bin")
        assert os.path.exists(bin_path)

    def test_export_ply(self, tmp_path) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())

        path = str(tmp_path / "output.ply")
        mesh.save_ply(path)
        assert os.path.exists(path)

    def test_export_obj(self, tmp_path) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())

        path = str(tmp_path / "output.obj")
        mesh.save_obj(path)
        assert os.path.exists(path)

    def test_all_formats_consistent(self, tmp_path) -> None:
        """三种格式导出的顶点数/面数一致."""
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())

        n_v = mesh.n_vertices
        n_f = mesh.n_faces

        # PLY.
        ply_path = str(tmp_path / "mesh.ply")
        mesh.save_ply(ply_path)
        loaded = TriangleMesh.load_ply(ply_path)
        assert loaded.n_vertices == n_v
        assert loaded.n_faces == n_f

        # OBJ.
        obj_path = str(tmp_path / "mesh.obj")
        mesh.save_obj(obj_path)
        with open(obj_path, encoding="utf-8") as f:
            obj_content = f.read()
        v_count = sum(1 for line in obj_content.splitlines() if line.startswith("v "))
        f_count = sum(1 for line in obj_content.splitlines() if line.startswith("f "))
        assert v_count == n_v
        assert f_count == n_f

        # glTF.
        gltf_path = str(tmp_path / "mesh.gltf")
        mesh.save_gltf(gltf_path)
        with open(gltf_path, encoding="utf-8") as f:
            gltf = json.load(f)
        assert gltf["accessors"][0]["count"] == n_v
        assert gltf["accessors"][1]["count"] == n_f * 3


class TestStatistics:
    """AC: 破洞率 + 三角面数可统计."""

    def test_face_count_stat(self) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())
        s = mesh.summary()
        assert "n_faces" in s
        assert s["n_faces"] > 0

    def test_hole_ratio_stat(self) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity())
        s = mesh.summary()
        assert "hole_ratio" in s
        assert 0.0 <= s["hole_ratio"] <= 1.0


class TestPLYRoundTrip:
    def test_save_load_roundtrip(self, tmp_path) -> None:
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        gen = MeshGenerator()
        mesh = gen.generate(dm, Pose.identity(), image=img)

        path = str(tmp_path / "roundtrip.ply")
        mesh.save_ply(path)
        loaded = TriangleMesh.load_ply(path)
        assert loaded.n_vertices == mesh.n_vertices
        assert loaded.n_faces == mesh.n_faces
        assert np.allclose(loaded.vertices, mesh.vertices, atol=1e-5)


class TestEndToEndWorkflow:
    """完整工作流: Image → Depth + VO → Mesh → 多格式导出."""

    def test_full_workflow(self, tmp_path) -> None:
        # 1. 生成图像.
        h, w = 8, 16
        frames = []
        for i in range(3):
            img = np.full((h, w, 3), 50 + i * 30, dtype=np.uint8)
            f = type("Frame", (), {})()
            f.image = img
            f.frame_id = i
            f.timestamp = float(i)
            f.depth = None
            f.pose = None
            frames.append(f)

        # 2. 深度 + VO.
        depth_est = DummyDepthEstimator()
        for f in frames:
            f.depth = depth_est.estimate(f.image)
        vo = DummyVisualOdometry(step_distance=0.5)
        traj = vo.estimate(frames)
        for f, pose in zip(frames, traj, strict=True):
            f.pose = pose

        # 3. 生成 mesh (第 0 帧).
        gen = MeshGenerator(MeshGeneratorConfig(projection="equirect"))
        mesh = gen.generate(frames[0].depth, frames[0].pose, image=frames[0].image)
        assert mesh.n_vertices > 0
        assert mesh.n_faces > 0
        assert not mesh.has_nan()
        assert mesh.has_vertex_colors

        # 4. 多格式导出.
        for ext in ["ply", "obj", "gltf"]:
            path = str(tmp_path / f"mesh.{ext}")
            if ext == "ply":
                mesh.save_ply(path)
            elif ext == "obj":
                mesh.save_obj(path)
            else:
                mesh.save_gltf(path)
            assert os.path.exists(path)

    def test_perspective_workflow(self, tmp_path) -> None:
        """透视投影完整流程."""
        est = DummyDepthEstimator()
        img = np.zeros((8, 16, 3), dtype=np.uint8)
        dm = est.estimate(img)
        K = np.array([[200, 0, 8], [0, 200, 4], [0, 0, 1]], dtype=np.float64)
        gen = MeshGenerator(MeshGeneratorConfig(projection="perspective"))
        mesh = gen.generate(dm, Pose.identity(), camera_matrix=K, image=img)
        assert mesh.n_vertices > 0
        assert mesh.n_faces > 0
        path = str(tmp_path / "persp.ply")
        mesh.save_ply(path)
        assert os.path.exists(path)
