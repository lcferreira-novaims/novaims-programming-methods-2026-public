import pandas as pd
import pandas.testing as pdt

from pm_labs.pipeline import (
    AddGrowthRateStage,
    DropMissingStage,
    ParseTimestampStage,
    Pipeline,
    RenameColumnsStage,
)


def test_drop_missing_stage_drops_rows_and_resets_index():
    frame = pd.DataFrame({"value": [1.0, None, 3.0]}, index=[4, 8, 12])

    result = DropMissingStage(["value"]).transform(frame)

    assert result["value"].tolist() == [1.0, 3.0]
    assert result.index.tolist() == [0, 1]


def test_rename_columns_stage_renames_requested_columns():
    frame = pd.DataFrame({"old": [1]})

    result = RenameColumnsStage({"old": "new"}).transform(frame)

    assert result.columns.tolist() == ["new"]
    assert frame.columns.tolist() == ["old"]


def test_parse_timestamp_stage_formats_months_as_strings():
    frame = pd.DataFrame({"timestamp": ["2024011509", "2024030100"]})

    result = ParseTimestampStage("timestamp").transform(frame)

    assert result["period"].tolist() == ["2024-01", "2024-03"]
    assert all(isinstance(period, str) for period in result["period"])
    assert "period" not in frame.columns


def test_add_growth_rate_sorts_by_group_and_date():
    frame = pd.DataFrame(
        {
            "group": ["a", "a", "b", "a"],
            "value": [30, 10, 8, 20],
            "date": ["2024-03", "2024-01", "2024-01", "2024-02"],
        }
    )

    result = AddGrowthRateStage("group", "value", "date").transform(frame)

    assert result[["group", "date"]].values.tolist() == [
        ["a", "2024-01"],
        ["a", "2024-02"],
        ["a", "2024-03"],
        ["b", "2024-01"],
    ]
    assert pd.isna(result.loc[0, "growth_rate"])
    assert result.loc[1, "growth_rate"] == 1.0
    assert result.loc[2, "growth_rate"] == 0.5
    assert pd.isna(result.loc[3, "growth_rate"])


def test_pipeline_runs_stages_in_order_without_mutating_source():
    frame = pd.DataFrame(
        {
            "article": ["a", "a", "a"],
            "views": [30, 10, 20],
            "timestamp": ["2024030100", "2024010100", "2024020100"],
        }
    )
    original = frame.copy(deep=True)
    pipeline = Pipeline()
    pipeline.add_stage(RenameColumnsStage({"timestamp": "raw_timestamp"}))
    pipeline.add_stage(ParseTimestampStage("raw_timestamp"))
    pipeline.add_stage(AddGrowthRateStage("article", "views", "period"))
    pipeline.add_stage(DropMissingStage(["growth_rate"]))

    result = pipeline.run(frame)

    assert result[["period", "growth_rate"]].values.tolist() == [
        ["2024-02", 1.0],
        ["2024-03", 0.5],
    ]
    pdt.assert_frame_equal(frame, original)
