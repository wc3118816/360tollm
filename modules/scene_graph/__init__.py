"""M026 Scene Graph 子包.

公开 API:
    SceneNode         - 场景图节点 (label + position + confidence)
    SceneEdge         - 场景图边 (relation + confidence + source)
    SceneGraph        - 场景图 (nodes + edges + serialize/deserialize)
    SceneGraphBuilder - 从 Object3DList 构建场景图
    GraphBuilderConfig - 构建配置
    RELATIONS         - 支持的关系类型集合

用法:
    from modules.scene_graph import SceneGraphBuilder
    builder = SceneGraphBuilder()
    graph = builder.build(object_list)
    json_str = graph.serialize()  # AC: 图结构可序列化
"""

from __future__ import annotations

from .builder import GraphBuilderConfig, SceneGraphBuilder
from .types import RELATIONS, SceneEdge, SceneGraph, SceneNode

__all__ = [
    "SceneNode",
    "SceneEdge",
    "SceneGraph",
    "SceneGraphBuilder",
    "GraphBuilderConfig",
    "RELATIONS",
]
