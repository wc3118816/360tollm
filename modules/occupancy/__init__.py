"""M018 Occupancy Map 子包.

公开 API:
    OccupancyGrid        - 三态占用栅格 (free/occupied/unknown + decay)
    OccupancyBuilder     - 点云 → 占用栅格 (射线投射)
    OccupancyBuilderConfig - 构建配置

    UNKNOWN / FREE / OCCUPIED - 三态枚举常量

用法:
    from modules.occupancy import OccupancyBuilder
    builder = OccupancyBuilder()
    grid = builder.build(points, sensor_origin=np.zeros(3))
    status = grid.status_at([1.0, 2.0, 3.0])
    grid.decay(0.95)
"""

from __future__ import annotations

from .builder import OccupancyBuilder, OccupancyBuilderConfig
from .types import FREE, OCCUPIED, UNKNOWN, OccupancyGrid

__all__ = [
    "OccupancyGrid",
    "OccupancyBuilder",
    "OccupancyBuilderConfig",
    "UNKNOWN",
    "FREE",
    "OCCUPIED",
]
