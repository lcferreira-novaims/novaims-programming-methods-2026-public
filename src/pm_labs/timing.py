"""Empirical timing helpers used by the Programming Methods labs."""

import statistics
import timeit
from collections.abc import Callable, Mapping, Sequence


def time_growth[Data](
    inputs: list[Data],
    operation: Callable[[Data], object],
    number: int = 200,
    repeats: int = 10,
) -> tuple[list[float], list[float]]:
    """Measure an operation on each explicitly prepared input.

    Each sample runs `operation` `number` times. Returns mean and population
    standard deviation in seconds per call.
    """
    means, stds = [], []
    for input_data in inputs:
        samples = timeit.repeat(
            lambda input_data=input_data: operation(input_data),
            number=number,
            repeat=repeats,
        )
        per_call = [sample / number for sample in samples]
        means.append(statistics.mean(per_call))
        stds.append(statistics.pstdev(per_call))
    return means, stds


def plot_growth(
    sizes: Sequence[int],
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    xlabel: str,
    title: str,
    ylabel: str = "seconds per call",
) -> None:
    """Plot timed series with standard-deviation error bars on log-log axes."""
    import matplotlib.pyplot as plt

    _, axis = plt.subplots()
    for label, (means, stds) in series.items():
        axis.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label=label)

    tick_sizes = sizes[::2]
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(tick_sizes, [f"{size:,}" for size in tick_sizes])
    axis.minorticks_off()
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    plt.show()
