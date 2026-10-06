import numpy as np

from pm_labs import schelling
from pm_labs.schelling import Agent, SchellingSimulation, SegregationGrid


def test_grid_initialization_has_requested_cell_counts():
    grid = SegregationGrid(size=4, empty_ratio=0.25, group_ratio=0.5, seed=1)
    cells = list(grid.grid.flat)

    assert sum(cell is None for cell in cells) == 4
    assert sum(cell is not None and cell.group == 0 for cell in cells) == 6
    assert sum(cell is not None and cell.group == 1 for cell in cells) == 6


def test_neighbors_returns_only_in_bounds_adjacent_cells():
    grid = SegregationGrid(size=3, empty_ratio=0, group_ratio=0.5, seed=1)

    assert len(grid.neighbors(1, 1)) == 8
    assert set(grid.neighbors(0, 0)) == {(0, 1), (1, 0), (1, 1)}


def test_happiness_uses_share_of_occupied_neighbors():
    grid = SegregationGrid(size=2, empty_ratio=0, group_ratio=0.5, seed=1)
    grid.grid = np.array(
        [[Agent(0), Agent(0)], [Agent(1), None]], dtype=object
    )

    assert grid.is_happy(0, 0, threshold=0.5)
    assert not grid.is_happy(0, 0, threshold=0.6)

    grid.grid[0, 1] = None
    grid.grid[1, 0] = None
    assert grid.is_happy(0, 0, threshold=1.0)


def test_step_moves_unhappy_agents_into_empty_cells(monkeypatch):
    grid = SegregationGrid(size=2, empty_ratio=0, group_ratio=0.5, seed=1)
    grid.grid = np.array(
        [[Agent(0), Agent(1)], [None, Agent(1)]], dtype=object
    )
    simulation = SchellingSimulation(grid, threshold=1.0)
    monkeypatch.setattr(schelling.random, "shuffle", lambda values: None)
    groups_before = sorted(
        cell.group for cell in grid.grid.flat if cell is not None
    )

    moves = simulation.step()

    groups_after = sorted(
        cell.group for cell in grid.grid.flat if cell is not None
    )
    assert moves == 1
    assert groups_after == groups_before
    assert sum(cell is None for cell in grid.grid.flat) == 1


def test_run_stops_when_no_agents_move():
    grid = SegregationGrid(size=1, empty_ratio=0, group_ratio=1.0, seed=1)
    simulation = SchellingSimulation(grid, threshold=1.0)

    assert simulation.run(max_steps=5) == [0]
