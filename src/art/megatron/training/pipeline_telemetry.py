from collections.abc import Sequence


def aggregate_pipeline_rank_metrics(
    rows: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Join one rank-local telemetry row per physical pipeline stage."""
    if not rows:
        raise ValueError("pipeline rank aggregation requires at least one row")
    pp_size = int(rows[0]["pipeline/pp_size"])
    if len(rows) != pp_size:
        raise RuntimeError(
            f"pipeline telemetry has {len(rows)} stages, expected {pp_size}"
        )
    by_stage = {int(row["pipeline/pp_rank"]): row for row in rows}
    if set(by_stage) != set(range(pp_size)):
        raise RuntimeError("pipeline telemetry has duplicate or missing PP ranks")
    out = {
        key: value
        for key, value in by_stage[0].items()
        if not key.startswith("pipeline/stage_")
        and key != "pipeline/stage_compute_imbalance_fraction"
    }
    for stage, row in sorted(by_stage.items()):
        prefix = f"pipeline/stage_{stage}/"
        stage_values = {
            key: value for key, value in row.items() if key.startswith(prefix)
        }
        if not stage_values:
            raise RuntimeError(f"pipeline telemetry is missing stage {stage} values")
        out.update(stage_values)
    stage_compute = [
        out[f"pipeline/stage_{stage}/forward_compute_s"]
        + out[f"pipeline/stage_{stage}/backward_compute_s"]
        for stage in range(pp_size)
    ]
    maximum = max(stage_compute, default=0.0)
    out["pipeline/stage_compute_imbalance_fraction"] = (
        (maximum - min(stage_compute)) / maximum if maximum else 0.0
    )
    return out
