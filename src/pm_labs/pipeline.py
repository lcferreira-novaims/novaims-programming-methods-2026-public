"""Data-transformation pipeline classes from Lab 4 (composition example)."""

from typing import Optional

import pandas as pd


class PipelineStage:
    """Base class for a single step in a data transformation pipeline."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class DropMissingStage(PipelineStage):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(subset=self.columns).reset_index(drop=True)


class RenameColumnsStage(PipelineStage):
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.mapping)


class ParseTimestampStage(PipelineStage):
    def __init__(self, timestamp_col: str, new_col: str = "period"):
        self.timestamp_col = timestamp_col
        self.new_col = new_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.new_col] = pd.to_datetime(
            df[self.timestamp_col], format="%Y%m%d%H"
        ).dt.strftime("%Y-%m")
        return df


class AddGrowthRateStage(PipelineStage):
    def __init__(self, group_col: str, value_col: str, date_col: str, new_col: str = "growth_rate"):
        self.group_col = group_col
        self.value_col = value_col
        self.date_col = date_col
        self.new_col = new_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Sort by group AND date - sorting by group alone does not
        # guarantee years are in order within each group, which would
        # make pct_change() compute nonsense.
        df = df.sort_values([self.group_col, self.date_col]).reset_index(drop=True)
        df[self.new_col] = df.groupby(self.group_col)[self.value_col].pct_change()
        return df


class Pipeline:
    def __init__(self, stages: Optional[list[PipelineStage]] = None):
        self.stages = stages or []

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        self.stages.append(stage)
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        for stage in self.stages:
            df = stage.transform(df)
        return df
