"""M048 MVP Demo: 360 相机空间智能最小闭环.

完整管线:
    Osmo 360 → Video → Frame → Object Detection → Depth
    → Camera Pose → 3D Object → World Model → LLM → 用户问题

MVP 使用 Dummy 后端 (无真实 360 视频 / 无真实 LLM):
- DummyDetector: 合成检测结果
- DummyDepthEstimator: 合成深度图
- IdentityPose: 相机在原点
- ObjectAssociator: 2D+Depth+Pose → 3D
- WorldModel + SpatialMemory + SceneGraphBuilder
- SpatialQA: 自然语言问答

必须支持的 4 个问题 (AC):
1. "我前面有什么？"
2. "左边是什么？"
3. "桌子离我多远？"
4. "刚才的人在哪里？"

AC: 四个问题均可完成闭环并提供结构化证据.

用法:
    demo = MVPDemo()
    demo.run()  # 处理多帧 + 回答 4 个问题
    # 或单次问答:
    answer = demo.ask("桌子离我多远？")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modules.depth import DummyDepthEstimator
from modules.detection import DummyDetector, DummyDetectorConfig
from modules.logging import get_logger
from modules.objects import ObjectAssociator
from modules.scene_graph import SceneGraph, SceneGraphBuilder
from modules.spatial_memory import SpatialMemory
from modules.spatial_qa import Answer, SpatialQA
from modules.vo.types import Pose
from modules.world_model import WorldModel

_LOG = get_logger("modules.demo.mvp")

_NDIM_3D = 3
# Demo 默认帧数.
_DEFAULT_N_FRAMES = 5
# Demo 默认图像尺寸.
_IMG_HEIGHT = 100
_IMG_WIDTH = 200
_IMG_CHANNELS = 3


@dataclass
class MVPDemo:
    """MVP Demo: 360 相机空间智能最小闭环.

    端到端管线:
        Frame → Detection + Depth + Pose → 3D Object
        → WorldModel + SceneGraph + SpatialMemory → SpatialQA

    Attributes:
        n_frames: 处理帧数.
        n_detections: 每帧检测数.
        detector: 目标检测器.
        depth_est: 深度估计器.
        associator: 2D→3D 关联器.
        world_model: 世界模型.
        spatial_memory: 空间记忆.
        sg_builder: 场景图构建器.
        qa: 空间问答 Agent.
    """

    n_frames: int = _DEFAULT_N_FRAMES
    n_detections: int = 3
    detector: DummyDetector = field(init=False)
    depth_est: DummyDepthEstimator = field(init=False)
    associator: ObjectAssociator = field(default_factory=ObjectAssociator)
    world_model: WorldModel = field(default_factory=WorldModel)
    spatial_memory: SpatialMemory = field(default_factory=SpatialMemory)
    sg_builder: SceneGraphBuilder = field(default_factory=SceneGraphBuilder)
    scene_graph: SceneGraph | None = field(default=None, repr=False)
    qa: SpatialQA = field(init=False)
    _processed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self._log = _LOG
        self.detector = DummyDetector(
            DummyDetectorConfig(n_detections=self.n_detections, latency_ms=0)
        )
        self.depth_est = DummyDepthEstimator()
        self.qa = SpatialQA(
            world_model=self.world_model,
            spatial_memory=self.spatial_memory,
        )

    def process_frames(self) -> None:
        """处理多帧, 构建 World Model + SpatialMemory."""
        img = np.zeros((_IMG_HEIGHT, _IMG_WIDTH, _IMG_CHANNELS), dtype=np.uint8)

        for i in range(self.n_frames):
            pose = Pose.identity(frame_id=i)
            depth_map = self.depth_est.estimate(img)
            detections = self.detector.detect(img, timestamp=float(i))
            objects_3d = self.associator.associate(detections, depth_map, pose)

            self.world_model.add_objects(objects_3d)
            self.scene_graph = self.sg_builder.build(objects_3d, existing_graph=self.scene_graph)
            self.spatial_memory.observe(objects_3d)

            self._log.info(
                "frame processed",
                frame_id=i,
                n_objects=self.world_model.n_objects,
                n_tracked=self.spatial_memory.n_objects,
            )

        # 更新 QA 的 world_model (因为 add_objects 修改了 model).
        self.qa.set_world_model(self.world_model)
        self.qa.set_spatial_memory(self.spatial_memory)
        self._processed = True

    def ask(self, question: str) -> Answer:
        """回答空间问题.

        Args:
            question: 自然语言问题.

        Returns:
            Answer 答案 + 结构化证据.
        """
        if not self._processed:
            self.process_frames()

        answer = self.qa.ask(question, reference_position=np.zeros(_NDIM_3D))
        self._log.info(
            "ask",
            question=question,
            query_type=answer.query_type,
            n_evidence=len(answer.evidence),
            confidence=answer.confidence,
        )
        return answer

    def ask_demo_questions(self) -> list[Answer]:
        """回答 4 个示例问题 (AC).

        Returns:
            list[Answer] 4 个答案.
        """
        questions = self.get_demo_questions()
        return [self.ask(q) for q in questions]

    @staticmethod
    def get_demo_questions() -> list[str]:
        """返回 4 个示例问题 (mission_list.md)."""
        return [
            "我前面有什么？",
            "左边是什么？",
            "桌子离我多远？",
            "刚才的人在哪里？",
        ]

    def run(self) -> dict[str, Any]:
        """运行完整 MVP Demo.

        Returns:
            dict 包含:
            - questions: 问题列表
            - answers: 答案列表
            - n_objects: 世界模型物体数
            - n_tracked: 空间记忆跟踪数
            - n_relations: 场景图关系数
        """
        self._log.info("MVP Demo starting...")
        self.process_frames()
        questions = self.get_demo_questions()
        answers = [self.ask(q) for q in questions]

        result: dict[str, Any] = {
            "questions": questions,
            "answers": [a.to_dict() for a in answers],
            "n_objects": self.world_model.n_objects,
            "n_tracked": self.spatial_memory.n_objects,
            "n_relations": (self.scene_graph.n_edges if self.scene_graph else 0),
        }
        self._log.info(
            "MVP Demo complete",
            n_objects=result["n_objects"],
            n_tracked=result["n_tracked"],
            n_answers=len(answers),
        )
        return result

    def print_results(self) -> None:
        """打印 MVP Demo 结果 (人类可读)."""
        result = self.run()
        print("\n" + "=" * 60)
        print("360 ToLLM MVP Demo — 空间智能最小闭环")
        print("=" * 60)
        print(f"帧数: {self.n_frames}")
        print(f"世界模型物体: {result['n_objects']}")
        print(f"空间记忆跟踪: {result['n_tracked']}")
        print(f"场景图关系: {result['n_relations']}")
        print("-" * 60)
        for q, a in zip(result["questions"], result["answers"], strict=False):
            print(f"\n问: {q}")
            print(f"答: {a['text']}")
            print(f"查询类型: {a['query_type']}")
            print(f"置信度: {a['confidence']}")
            if a["evidence"]:
                print(f"证据: {a['evidence']}")
        print("\n" + "=" * 60)
