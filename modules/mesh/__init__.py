"""M017 Mesh 重建子包.

公开 API:
    TriangleMesh        - 三角网格数据结构 (vertices + faces + PLY/OBJ/glTF)
    MeshGenerator       - 深度图 → 三角网格
    MeshGeneratorConfig - 生成器配置

用法:
    from modules.mesh import MeshGenerator, MeshGeneratorConfig
    gen = MeshGenerator()
    mesh = gen.generate(depth_map, pose)
    mesh.save_gltf("output.gltf")
    mesh.save_obj("output.obj")
    mesh.save_ply("output.ply")
"""

from __future__ import annotations

from .generator import MeshGenerator, MeshGeneratorConfig
from .types import TriangleMesh

__all__ = [
    "TriangleMesh",
    "MeshGenerator",
    "MeshGeneratorConfig",
]
