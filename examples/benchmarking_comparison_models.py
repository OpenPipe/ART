import art
import asyncio
from pydantic import BaseModel
import litellm
import re
from litellm.types.utils import Choices
from art.utils.litellm_utils import convert_litellm_choice_to_openai
from art.utils import iterate_dataset


PROJECT_NAME = "benchmarking-comparison-models"


# This is a project-specific definition of a scenario you might use for training
class MyScenarioSpec(BaseModel):
    input_1: int
    input_2: int
    output: int


training_scenarios = [
    MyScenarioSpec(input_1=1, input_2=2, output=3),
    MyScenarioSpec(input_1=2, input_2=3, output=5),
    MyScenarioSpec(input_1=3, input_2=4, output=7),
]

test_scenarios = [
    MyScenarioSpec(input_1=4, input_2=5, output=9),
    MyScenarioSpec(input_1=5, input_2=6, output=11),
]


# Projects can define whatever they need to inside their config objects. In this
# case, we're storing a `litellm_model_name`, which we can use to point
# inference to a specific model on litellm, as well as a `use_thinking` flag,
# which lets us compare models trained (or prompted) to use CoT to models
# trained to just output the answer directly in the same project. Our rollout
# and reward functions will use this config to adjust their behavior, but to ART
# itself it's completely opaque.
class MyConfig(BaseModel):
    litellm_model_name: str | None = None
    use_thinking: bool = False


# Dummy reward function that simply checks whether the model's answer is
# correct. It supports both thinking and non-thinking models.
def reward(config: MyConfig, scenario: MyScenarioSpec, response: str):
    if config.use_thinking:
        answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer is None:
            return -1
        response = answer.group(1).strip()
    parsed_response = int(response)
    if parsed_response == scenario.output:
        return 1
    else:
        return -1


# This function runs our rollout against our test scenarios, and logs the results.
# We can use it to benchmark both trainable and prompted models.
async def benchmark_model(model: art.Model):
    trajectories = await art.gather_trajectories(
        (rollout(model, scenario) for scenario in test_scenarios),
        pbar_desc="benchmark",
        max_exceptions=100,
    )
    valid_trajectories = [t for t in trajectories if isinstance(t, art.Trajectory)]
    await model.log(valid_trajectories)


# While ART gives you the flexibility to define your rollouts however you want
# and pass whatever arguments they need, in general I like the pattern of just
# having them take a `model` and a `scenario` and returning a `Trajectory`. Any
# fields that may affect rollout behavior can be stored on the model's config
# object, which gets logged and stored by the system so you can remember how you
# configured things for a specific model later.


# In this case we have a very simple rollout function, but you can see how we're
# using the `use_thinking` flag to change the prompt slightly.
async def rollout(model: art.Model, scenario: MyScenarioSpec) -> art.Trajectory:
    assert isinstance(model.config, MyConfig)
    messages = []
    if model.config.use_thinking:
        messages.append(
            {
                "role": "system",
                "content": "You'll be given a math problem by the user. Before answering, first output your thinking inside <thinking> tags. Then, output your answer inside <answer> tags.",
            }
        )

    messages.append(
        {"role": "user", "content": f"What is {scenario.input_1} + {scenario.input_2}?"}
    )

    response = await litellm.acompletion(
        model=model.config.litellm_model_name or f"hosted_vllm/{model.name}",
        messages=messages,
        # We can use the built-in LiteLLM caching so if we re-run the same
        # rollout for our comparison models we don't need to run inference
        # again. However, for our trainable models it's very important we don't
        # cache so we get diversity in our rollouts.
        caching=not model.trainable,
        api_key=model.api_key,
        base_url=model.base_url,
    )
    choice: Choices = response.choices[0]  # type: ignore
    response_text: str = choice.message.content  # type: ignore

    messages.append(convert_litellm_choice_to_openai(choice))

    # Your rollout function should always return a Trajectory object.
    return art.Trajectory(
        messages_and_choices=messages,
        reward=reward(model.config, scenario, response_text),
    )


async def train_model(model: art.TrainableModel):
    train_iterator = iterate_dataset(
        training_scenarios,
        batch_size=4,
        initial_step=await model.get_step(),
    )

    for batch, epoch, global_step, epoch_step in train_iterator:
        groups = await art.gather_trajectory_groups(
            art.TrajectoryGroup(
                (rollout(model, scenario) for _ in range(6)),
            )
            for scenario in batch
        )
        await model.train(groups)

        if global_step % 20 == 0:
            # Every 20 steps let's benchmark our model under training so we can
            # see how it's doing.
            await benchmark_model(model)

    # At the end of training, let's benchmark the model again to see where it
    # ended up.
    await benchmark_model(model)


async def main():
    # Here we define a list of prompted comparison models we want to assess
    # performance for.
    gpt_4o = art.Model(
        name="gpt-4o",
        project=PROJECT_NAME,
        config=MyConfig(litellm_model_name="openai/gpt-4o"),
    )
    gpt_4o_thinking = gpt_4o.model_copy(deep=True)
    assert isinstance(gpt_4o_thinking.config, MyConfig)
    gpt_4o_thinking.name = "gpt-4o-thinking"

    gemini_flash = gpt_4o.model_copy(deep=True)
    assert isinstance(gemini_flash.config, MyConfig)
    gemini_flash.name = "gemini-flash-1.5"
    gemini_flash.config.litellm_model_name = "gemini/gemini-flash-1.5"

    gemini_flash_thinking = gemini_flash.model_copy(deep=True)
    assert isinstance(gemini_flash_thinking.config, MyConfig)
    gemini_flash_thinking.name = "gemini-flash-1.5-thinking"

    prompted_models: list[art.Model] = [
        gpt_4o,
        gpt_4o_thinking,
        gemini_flash,
        gemini_flash_thinking,
    ]

    # We also can define the models we want to train.
    qwen = art.TrainableModel(
        name="qwen-2.5-14b-instruct",
        project="my-project",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=MyConfig(),
    )
    qwen_thinking = qwen.model_copy(deep=True)
    assert isinstance(qwen_thinking.config, MyConfig)
    qwen_thinking.name = "qwen-2.5-14b-instruct-thinking"

    train_models: list[art.TrainableModel] = [
        qwen,
        qwen_thinking,
    ]

    # We need to register all our models with the local API so they're
    # available for training and logging.
    api = art.LocalAPI()
    for model in train_models + prompted_models:
        await model.register(api)

    # For prompted models, we can benchmark them right away.
    for model in prompted_models:
        await benchmark_model(model)

    # For trainable models, we need to train them. Benchmarking will happen
    # automatically throughout training.
    for model in train_models:
        await train_model(model)


if __name__ == "__main__":
    asyncio.run(main())
