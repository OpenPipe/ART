import weave
from art_e.rollout import ProjectTrajectory
from weave.trace.autopatch import AutopatchSettings

import art


def report_trajectory(
    model: art.Model,
    trajectory: ProjectTrajectory,
    step: int = 0,
):
    client = weave.init(
        model.project, autopatch_settings=AutopatchSettings(disable_autopatch=True)
    )

    inputs = {
        "model": model.name,
        "scenario": trajectory.scenario,
        "step": step,
    }

    if isinstance(model, art.TrainableModel):
        inputs["base_model"] = model.base_model

    call = client.create_call(
        "trajectory",
        inputs=inputs,
    )
    client.finish_call(call, output={"tr": trajectory})
