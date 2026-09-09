"""M018 单元测试: OccupancyBuilder 点云 → 占用栅格.

覆盖:
- 基本构建 (点标记 occupied)
- 射线投射标记 free
- sensor_origin 位置查询
- 增量更新 (existing grid)
- 空点云
- decay 整合
- metadata
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.occupancy import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyBuilder,
    OccupancyBuilderConfig,
)


class TestBuild:
    def test_basic_build(self) -> None:
        """点云 → occupied 体素."""
        points = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        builder = OccupancyBuilder(OccupancyBuilderConfig(voxel_size=0.5))
        grid = builder.build(points, sensor_origin=np.zeros(3))
        # 点应标记为 occupied.
        assert grid.status_at(np.array([1.0, 0.0, 0.0])) == OCCUPIED
        assert grid.metadata["n_updates"] == 1

    def test_point_marked_occupied(self) -> None:
        points = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        builder = OccupancyBuilder()
        grid = builder.build(points, sensor_origin=np.zeros(3))
        assert grid.is_occupied(np.array([5.0, 0.0, 0.0]))

    def test_ray_casting_free(self) -> None:
        """射线投射: sensor → point 途经体素应标记 free."""
        points = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        builder = OccupancyBuilder(OccupancyBuilderConfig(voxel_size=0.5))
        grid = builder.build(points, sensor_origin=np.zeros(3))
        # sensor 原点附近应 free.
        assert grid.status_at(np.array([0.5, 0.0, 0.0])) == FREE
        # 中间也应 free.
        assert grid.status_at(np.array([2.0, 0.0, 0.0])) == FREE
        # 终点 occupied.
        assert grid.status_at(np.array([5.0, 0.0, 0.0])) == OCCUPIED

    def test_sensor_origin_in_grid(self) -> None:
        """sensor_origin 应在 grid 范围内."""
        points = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        builder = OccupancyBuilder()
        grid = builder.build(points, sensor_origin=np.array([0.0, 0.0, 0.0]))
        idx = grid.world_to_voxel(np.zeros(3))
        assert np.all(idx >= 0)

    def test_empty_points(self) -> None:
        """空点云 → 全 UNKNOWN."""
        builder = OccupancyBuilder()
        grid = builder.build(np.zeros((0, 3), dtype=np.float32), sensor_origin=np.zeros(3))
        assert grid.n_voxels > 0
        # 全 UNKNOWN.
        assert np.all(grid.grid == UNKNOWN)

    def test_invalid_points_shape(self) -> None:
        builder = OccupancyBuilder()
        with pytest.raises(ValueError, match="points"):
            builder.build(np.zeros((5, 2), dtype=np.float32), sensor_origin=np.zeros(3))

    def test_invalid_sensor_origin(self) -> None:
        builder = OccupancyBuilder()
        with pytest.raises(ValueError, match="sensor_origin"):
            builder.build(np.zeros((1, 3), dtype=np.float32), sensor_origin=np.zeros(2))


class TestIncrementalUpdate:
    def test_update_existing(self) -> None:
        """增量更新: 已有 grid 加新点."""
        # 第一次构建 (大 padding 确保 grid 覆盖两个点).
        points1 = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
        builder = OccupancyBuilder(OccupancyBuilderConfig(voxel_size=0.5, padding=5.0))
        grid = builder.build(points1, sensor_origin=np.zeros(3))
        # 第二次更新 (加一个新点).
        points2 = np.array([[0.0, 3.0, 0.0]], dtype=np.float32)
        grid = builder.build(points2, sensor_origin=np.zeros(3), existing=grid)
        # 两个点都应 occupied.
        assert grid.is_occupied(np.array([3.0, 0.0, 0.0]))
        assert grid.is_occupied(np.array([0.0, 3.0, 0.0]))
        assert grid.metadata["n_updates"] == 2


class TestDecay:
    def test_decay_after_build(self) -> None:
        """构建后 decay → confidence 降低."""
        points = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        builder = OccupancyBuilder()
        grid = builder.build(points, sensor_origin=np.zeros(3))
        conf_before = grid.confidence.copy()
        grid.decay(decay_rate=0.5)
        # 有值的 confidence 应降低.
        occupied_mask = grid.grid == OCCUPIED
        if occupied_mask.any():
            assert (grid.confidence[occupied_mask] <= conf_before[occupied_mask]).all()

    def test_decay_to_unknown_full(self) -> None:
        """多次 decay → 所有 occupied → unknown."""
        points = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        builder = OccupancyBuilder()
        grid = builder.build(points, sensor_origin=np.zeros(3))
        # 多次衰减 (rate=0.1, threshold=0.5).
        for _ in range(20):
            grid.decay(decay_rate=0.1, conf_threshold=0.5)
        # 应全部变 UNKNOWN.
        assert grid.summary()["n_occupied"] == 0


class TestPositionQuery:
    """AC: 可查询任意位置占用状态."""

    def test_query_any_position(self) -> None:
        points = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
        builder = OccupancyBuilder(OccupancyBuilderConfig(voxel_size=0.5))
        grid = builder.build(points, sensor_origin=np.zeros(3))
        # 查询占用点.
        assert grid.status_at(np.array([3.0, 0.0, 0.0])) == OCCUPIED
        # 查询 free 点 (射线路径).
        assert grid.status_at(np.array([1.0, 0.0, 0.0])) == FREE
        # 查询超出范围点.
        assert grid.status_at(np.array([100.0, 100.0, 100.0])) == UNKNOWN
