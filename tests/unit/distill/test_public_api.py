import art
from art import distill
from art.distill.artifact import PreparedTrainingBatch
from art.distill.capture import (
    captured_context,
    generations,
    prepend_message,
)
from art.distill.prepare import all_generations, prepare, same_context
from art.distill.preparer import (
    AllCapturedGenerations,
    FailurePolicy,
    SameContext,
    mask_failed_generation,
    strict,
)
from art.distill.types import (
    AnyRevision,
    CapturedGeneration,
    CurrentStep,
    Example,
    ForwardKL,
    Frozen,
    GenerationPart,
    Loss,
    StudentOnPolicy,
    TeacherRevision,
    TopK,
    TrainingObjectives,
)


def test_top_level_distillation_exports() -> None:
    assert art.distill is distill
    assert art.CapturedGeneration is CapturedGeneration
    assert art.PreparedTrainingBatch is PreparedTrainingBatch
    assert art.TrainingObjectives is TrainingObjectives


def test_distill_guide_exports() -> None:
    expected = {
        "all_generations": all_generations,
        "AllCapturedGenerations": AllCapturedGenerations,
        "AnyRevision": AnyRevision,
        "captured_context": captured_context,
        "CurrentStep": CurrentStep,
        "Example": Example,
        "FailurePolicy": FailurePolicy,
        "ForwardKL": ForwardKL,
        "Frozen": Frozen,
        "GenerationPart": GenerationPart,
        "generations": generations,
        "Loss": Loss,
        "mask_failed_generation": mask_failed_generation,
        "prepare": prepare,
        "PreparedTrainingBatch": PreparedTrainingBatch,
        "prepend_message": prepend_message,
        "same_context": same_context,
        "SameContext": SameContext,
        "strict": strict,
        "StudentOnPolicy": StudentOnPolicy,
        "TeacherRevision": TeacherRevision,
        "TopK": TopK,
        "TrainingObjectives": TrainingObjectives,
    }

    for name, value in expected.items():
        assert getattr(distill, name) is value
