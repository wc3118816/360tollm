"""M018 单元测试: OccupancyGrid 数据结构.

覆盖:
- 构造 + 校验 (shape / origin / voxel_size)
- world_to_voxel / voxel_to_world 坐标变换
- status_at 位置查询
- is_occupied / is_free / is_unknown
- decay 时间衰减
- summary 统计
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyGrid


def _make_grid(
    shape: tuple[int, int, int] = (5, 5, 5),
    voxel_size: float = 0.5,
    origin: np.ndarray | None = None,
) -> OccupancyGrid:
    if origin is None:
        origin = np.zeros(3, dtype=np.float64)
    grid = np.zeros(shape, dtype=np.int8)
    conf = np.zeros(shape, dtype=np.float32)
    return OccupancyGrid(grid=grid, confidence=conf, origin=origin, voxel_size=voxel_size)


class TestConstruction:
    def test_basic_construction(self) -> None:
        grid = _make_grid()
        assert grid.shape == (5, 5, 5)
        assert grid.voxel_size == 0.5

    def test_invalid_grid_dim(self) -> None:
        with pytest.raises(ValueError, match="grid must be 3D"):
            OccupancyGrid(
                grid=np.zeros((5, 5), dtype=np.int8),
                confidence=np.zeros((5, 5), dtype=np.float32),
                origin=np.zeros(3),
                voxel_size=0.5,
            )

    def test_confidence_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="confidence shape"):
            OccupancyGrid(
                grid=np.zeros((5, 5, 5), dtype=np.int8),
                confidence=np.zeros((3, 3, 3), dtype=np.float32),
                origin=np.zeros(3),
                voxel_size=0.5,
            )

    def test_invalid_voxel_size(self) -> None:
        with pytest.raises(ValueError, match="voxel_size"):
            OccupancyGrid(
                grid=np.zeros((5, 5, 5), dtype=np.int8),
                confidence=np.zeros((5, 5, 5), dtype=np.float32),
                origin=np.zeros(3),
                voxel_size=0.0,
            )

    def test_origin_non_3d_raises(self) -> None:
        with pytest.raises(ValueError, match="origin"):
            OccupancyGrid(
                grid=np.zeros((5, 5, 5), dtype=np.int8),
                confidence=np.zeros((5, 5, 5), dtype=np.float32),
                origin=np.zeros(2),
                voxel_size=0.5,
            )


class TestCoordinateTransform:
    def test_world_to_voxel(self) -> None:
        grid = _make_grid(shape=(5, 5, 5), voxel_size=1.0, origin=np.zeros(3))
        idx = grid.world_to_voxel(np.array([2.5, 1.0, 0.5]))
        # floor((2.5, 1.0, 0.5) / 1.0) = (2, 1, 0).
        assert idx[0] == 2
        assert idx[1] == 1
        assert idx[2] == 0

    def test_world_to_voxel_batch(self) -> None:
        grid = _make_grid(shape=(5, 5, 5), voxel_size=1.0, origin=np.zeros(3))
        pts = np.array([[2.5, 1.0, 0.5], [0.0, 0.0, 0.0]])
        idx = grid.world_to_voxel(pts)
        assert idx.shape == (2, 3)
        assert idx[0, 0] == 2

    def test_world_to_voxel_out_of_range(self) -> None:
        grid = _make_grid(shape=(5, 5, 5), voxel_size=1.0, origin=np.zeros(3))
        # 超出范围 → -1.
        idx = grid.world_to_voxel(np.array([100.0, 100.0, 100.0]))
        assert np.all(idx == -1)

    def test_voxel_to_world_center(self) -> None:
        grid = _make_grid(shape=(5, 5, 5), voxel_size=1.0, origin=np.zeros(3))
        # voxel (0,0,0) 中心 = (0.5, 0.5, 0.5).
        world = grid.voxel_to_world(np.array([0, 0, 0]))
        assert np.allclose(world, [0.5, 0.5, 0.5])

    def test_round_trip(self) -> None:
        grid = _make_grid(shape=(10, 10, 10), voxel_size=0.5, origin=np.array([1.0, 2.0, 3.0]))
        world = np.array([2.3, 3.1, 4.2])
        idx = grid.world_to_voxel(world)
        world_back = grid.voxel_to_world(idx)
        # 应在同一个体素内 (差距 < voxel_size).
        assert np.linalg.norm(world - world_back) < grid.voxel_size


class TestStatusQuery:
    def test_status_at_unknown(self) -> None:
        grid = _make_grid()
        # 默认全 UNKNOWN.
        assert grid.status_at(np.array([0.1, 0.1, 0.1])) == UNKNOWN

    def test_status_at_occupied(self) -> None:
        grid = _make_grid()
        grid.grid[0, 0, 0] = OCCUPIED
        assert grid.status_at(np.array([0.1, 0.1, 0.1])) == OCCUPIED

    def test_status_at_free(self) -> None:
        grid = _make_grid()
        grid.grid[0, 0, 0] = FREE
        assert grid.status_at(np.array([0.1, 0.1, 0.1])) == FREE

    def test_status_out_of_range(self) -> None:
        grid = _make_grid()
        assert grid.status_at(np.array([100.0, 100.0, 100.0])) == UNKNOWN

    def test_is_occupied(self) -> None:
        grid = _make_grid()
        grid.grid[0, 0, 0] = OCCUPIED
        assert grid.is_occupied(np.array([0.1, 0.1, 0.1]))
        assert not grid.is_free(np.array([0.1, 0.1, 0.1]))
        assert not grid.is_unknown(np.array([0.1, 0.1, 0.1]))


class TestDecay:
    def test_decay_reduces_confidence(self) -> None:
        grid = _make_grid()
        grid.confidence[0, 0, 0] = 1.0
        grid.grid[0, 0, 0] = OCCUPIED
        grid.decay(decay_rate=0.5)
        assert grid.confidence[0, 0, 0] == 0.5
        # confidence > 阈值, 仍 OCCUPIED.
        assert grid.grid[0, 0, 0] == OCCUPIED

    def test_decay_to_unknown(self) -> None:
        grid = _make_grid()
        grid.confidence[0, 0, 0] = 0.1
        grid.grid[0, 0, 0] = OCCUPIED
        # 衰减后 < 阈值 → UNKNOWN.
        grid.decay(decay_rate=0.5, conf_threshold=0.1)
        assert grid.grid[0, 0, 0] == UNKNOWN
        assert grid.confidence[0, 0, 0] < 0.1

    def test_decay_count(self) -> None:
        grid = _make_grid()
        grid.decay()
        grid.decay()
        assert grid.metadata["decay_count"] == 2


class TestSummary:
    def test_summary(self) -> None:
        grid = _make_grid(shape=(3, 3, 3))
        grid.grid[0, 0, 0] = OCCUPIED
        grid.grid[1, 1, 1] = FREE
        s = grid.summary()
        assert s["shape"] == [3, 3, 3]
        assert s["n_unknown"] == 25  # 27 - 2
        assert s["n_free"] == 1
        assert s["n_occupied"] == 1
        assert s["n_voxels"] == 27
