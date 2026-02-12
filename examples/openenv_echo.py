# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openenv-core==0.1.13",
#     "openpipe-art==0.5.1",
# ]
#
# ///
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from envs.echo_env import EchoAction, EchoEnv
import weave

import art
from art.serverless.backend import ServerlessBackend

PROMPT = "Use at most 100 tokens; maximize the total character length of the output."
NUM_STEPS = 50
ROLLOUTS_PER_GROUP = 4


# In ART, the rollout function
async def rollout(model: art.TrainableModel, env_client: EchoEnv) -> art.Trajectory:
    # For the simple echo environment there's no internal state to reset, but we show resetting anyway to demonstrate the pattern.
    await asyncio.to_thread(env_client.reset)

    # We create an art.Trajectory object to store our messages as well as the final reward.
    traj = art.Trajectory(
        messages_and_choices=[{"role": "system", "content": PROMPT}], reward=0.0
    )

    # We use the model we're training to generate the next action to send to the environment. For this simple echo environment, the action is a single message.
    choice = (
        await model.openai_client().chat.completions.create(
            model=model.inference_model_name,
            messages=traj.messages(),
            max_completion_tokens=100,
            timeout=30,
        )
    ).choices[0]
    reply = (choice.message.content or "").strip()

    # We send the action to the environment.
    result = await asyncio.to_thread(env_client.step, EchoAction(message=reply))

    # We need to record the actual message we produced so we can use it for training later.
    traj.messages_and_choices.append(choice)

    # The environment gives us back a reward (in this case it's simply the length of the message we sent divided by 10). We record it so we can use it for training later.
    traj.reward = result.reward

    # We return the completed trajectory to the trainer.
    return traj.finish()


async def main() -> None:
    load_dotenv()

    weave.init("openenv-demo")

    # The ServerlessBackend requires a `WANDB_API_KEY` environment variable to be set. You can also use the ART `LocalBackend` to train on a local GPU.
    backend = ServerlessBackend()

    # We define a model that we'll train. The model is a LoRA adapter on top of Qwen3-14B.
    model = art.TrainableModel(
        name=f"openenv-echo-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}",
        project="openenv-demo",
        base_model="OpenPipe/Qwen3-14B-Instruct",
    )
    await model.register(backend)

    # We create a shared pool of environment clients for training, to avoid starting up and tearing down docker containers for each rollout.
    env_pool = [
        EchoEnv.from_docker_image("quixote13/echo-env:latest")
        for _ in range(ROLLOUTS_PER_GROUP)
    ]

    # We train the model for a fixed number of steps.
    for _step in range(await model.get_step(), NUM_STEPS):
        print(f"Gathering groups for step {_step}")

        # We
        groups = await art.gather_trajectory_groups(
            [art.TrajectoryGroup(rollout(model, env_client) for env_client in env_pool)]
        )

        result = await backend.train(model, groups)
        await model.log(groups, metrics=result.metrics, step=result.step, split="train")


asyncio.run(main())
