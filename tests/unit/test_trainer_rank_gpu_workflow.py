from pathlib import Path


def test_ready_for_review_reuse_has_stable_source_provenance() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/trainer-rank-gpu.yml"
    ).read_text()

    assert "github.event.action == 'ready_for_review'" in workflow
    assert "pull?.head?.repo?.full_name" in workflow
    assert "pull?.head?.ref" in workflow
    assert "pull?.head?.sha" in workflow
    assert "pull?.base?.sha" in workflow
    assert "identity.every(value =>" in workflow
    assert "typeof value === 'string' && value.length > 0" in workflow
    assert "run.head_repository?.full_name === pull.head.repo.full_name" in workflow
    assert "run.head_branch === pull.head.ref" in workflow
    assert "run.head_sha === pull.head.sha" in workflow
    assert "run.path === '.github/workflows/trainer-rank-gpu.yml'" in workflow
    assert "run.pull_requests" not in workflow


def test_ready_for_review_reuse_requires_exact_base_artifact_and_gpu_job() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/trainer-rank-gpu.yml"
    ).read_text()

    assert "job.name === 'Run on 2x H200' && job.conclusion === 'success'" in workflow
    assert "!artifact.expired" in workflow
    assert (
        "`trainer-rank-result-${run.id}-${run.run_attempt}-${pull.base.sha}`"
        in workflow
    )
