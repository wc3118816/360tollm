"""M011 Dummy Visual Odometry: 合成轨迹基线, 无 ML 依赖.

生成确定性合成轨迹:
- 相机沿 +x (forward) 匀速移动, 每帧位移 step_distance 米.
- 可选小幅 yaw 摆动 (模拟行人走路).
- 输出 Pose 序列 (4x4 SE3, world_to_camera).

用途:
1. 无 OpenCV / 无 ML 环境下的流水线测试.
2. M015 点云生成 / M021 2D→3D 的占位 VO.
3. 验证 VisualOdometry ABC + Trajectory + VOStats 完整链路.

注意: 输出是合成位姿, 不反映真实运动. 真实场景请用 FeatureVO (ORB).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from modules.projection.rotation import rotation_yaw

from .base import VisualOdometry
from .types import Pose

# 默认每帧位移 (米).
_DEFAULT_STEP = 0.1
# 默认 yaw 摆动幅度 (rad).
_DEFAULT_YAW_AMP = 0.0


class DummyVisualOdometry(VisualOdometry):
    """合成 VO 基线.

    生成沿 +x 匀速直线的轨迹, 可选 yaw 摆动.
    相同输入帧数 → 相同输出 (确定性).
    """

    backend_name = "dummy"

    def __init__(
        self,
        device: str = "cpu",
        step_distance: float = _DEFAULT_STEP,
        yaw_amplitude: float = _DEFAULT_YAW_AMP,
        yaw_frequency: float = 0.1,
    ) -> None:
        super().__init__(device=device)
        self.step_distance = step_distance
        self.yaw_amplitude = yaw_amplitude
        self.yaw_frequency = yaw_frequency

    def _estimate_poses(self, frames: Sequence[Any]) -> list[Pose | None]:
        """生成合成轨迹."""
        poses: list[Pose | None] = []
        for i, frame in enumerate(frames):
            timestamp = getattr(frame, "timestamp", float(i))
            frame_id = getattr(frame, "frame_id", i)

            # 相机在 world 坐标系的位置 (沿 +x 匀速).
            x = i * self.step_distance
            # yaw 摆动 (模拟走路).
            yaw = self.yaw_amplitude * np.sin(2 * np.pi * self.yaw_frequency * i)

            # world_to_camera = T_cam_world = (R_yaw(yaw) | -R_yaw(yaw) @ pos_world)
            R = rotation_yaw(yaw)
            pos_world = np.array([x, 0.0, 0.0], dtype=np.float64)
            t = -R @ pos_world

            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = R
            matrix[:3, 3] = t

            poses.append(
                Pose(
                    matrix=matrix,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    confidence=1.0,
                    metadata={
                        "synthetic": True,
                        "step_distance": self.step_distance,
                        "yaw": float(yaw),
                    },
                )
            )
        return poses

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(
            {
                "step_distance": self.step_distance,
                "yaw_amplitude": self.yaw_amplitude,
                "synthetic": True,
            }
        )
        return d
