"""Schelling's model of segregation classes from Lab 4."""

import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class Agent:
    def __init__(self, group: int):
        self.group = group

    def __repr__(self) -> str:
        return f"Agent(group={self.group})"


class SegregationGrid:
    def __init__(self, size: int, empty_ratio: float, group_ratio: float, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self.size = size
        self.grid = np.empty((size, size), dtype=object)

        n_cells = size * size
        n_empty = int(n_cells * empty_ratio)
        n_occupied = n_cells - n_empty
        n_group0 = int(n_occupied * group_ratio)
        n_group1 = n_occupied - n_group0

        cells = (
            [None] * n_empty
            + [Agent(0) for _ in range(n_group0)]
            + [Agent(1) for _ in range(n_group1)]
        )
        random.shuffle(cells)

        for i, cell in enumerate(cells):
            row, col = divmod(i, size)
            self.grid[row, col] = cell

    def neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        coords = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.size and 0 <= c < self.size:
                    coords.append((r, c))
        return coords

    def empty_cells(self) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.grid[r, c] is None
        ]

    def is_happy(self, row: int, col: int, threshold: float) -> bool:
        agent = self.grid[row, col]
        occupied = 0
        same_group = 0
        for r, c in self.neighbors(row, col):
            neighbor = self.grid[r, c]
            if neighbor is not None:
                occupied += 1
                if neighbor.group == agent.group:
                    same_group += 1

        if occupied == 0:
            return True
        return (same_group / occupied) >= threshold


class SchellingSimulation:
    def __init__(self, grid: SegregationGrid, threshold: float):
        self.grid = grid
        self.threshold = threshold

    def step(self) -> int:
        unhappy = [
            (r, c)
            for r in range(self.grid.size)
            for c in range(self.grid.size)
            if self.grid.grid[r, c] is not None
            and not self.grid.is_happy(r, c, self.threshold)
        ]
        random.shuffle(unhappy)

        available = self.grid.empty_cells()
        random.shuffle(available)

        moves = 0
        for (row, col) in unhappy:
            if not available:
                break
            new_row, new_col = available.pop()
            self.grid.grid[new_row, new_col] = self.grid.grid[row, col]
            self.grid.grid[row, col] = None
            moves += 1

        return moves

    def run(self, max_steps: int) -> list[int]:
        history = []
        for _ in range(max_steps):
            moved = self.step()
            history.append(moved)
            if moved == 0:
                break
        return history


def plot_grid(grid: "SegregationGrid", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    display = np.full((grid.size, grid.size), -1, dtype=int)
    for r in range(grid.size):
        for c in range(grid.size):
            agent = grid.grid[r, c]
            if agent is not None:
                display[r, c] = agent.group

    ax.imshow(display, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
