import asyncio
import os
from dotenv import load_dotenv

import unsloth  # Ensure Unsloth patches are applied early
import openai

import art
from art.local import LocalBackend


async def rollout(client: openai.AsyncOpenAI, prompt: str) -> art.Trajectory:
	messages: art.Messages = [
		{
			"role": "user",
			"content": prompt,
		}
	]
	chat_completion = await client.chat.completions.create(
		messages=messages, model=model.name, max_tokens=100, timeout=100
	)
	choice = chat_completion.choices[0]
	content = choice.message.content
	assert isinstance(content, str)
	if content == "yes":
		reward = 0.5
	elif content == "no":
		reward = 0.75
	elif content == "maybe":
		reward = 1.0
	else:
		reward = 0.0
	return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


def with_quotes(w: str) -> str:
	return f"'{w}'"


async def main() -> None:
	print("[script] Loading env...")
	load_dotenv()
	backend = LocalBackend()
	print("[script] Creating model...")
	base_model = os.getenv("ART_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
	print(f"[script] Using base_model={base_model}")
	global model
	model = art.TrainableModel(
		name="009",
		project="yes-no-maybe",
		base_model=base_model,
		_internal_config=art.dev.InternalModelConfig(
			_decouple_vllm_and_unsloth=True,
			engine_args=art.dev.EngineArgs(gpu_memory_utilization=0.12, max_model_len=4096, enable_sleep_mode=False),
		),
	)
	print("[script] Registering model...")
	await model.register(backend)
	print("[script] Model registered. Starting rollout/train loop...")

	prompts = [
		f"{prefix} with {', '.join([with_quotes(w) if use_quotes else w for w in words]) if len(words) == 3 else f'{words[0]}' + (f' or {words[1]}' if len(words) > 1 else '')}"
		for prefix in ["respond", "just respond"]
		for use_quotes in [True, False]
		for words in [
			["yes", "no", "maybe"],
			["maybe", "yes", "no"],
			["no", "yes", "maybe"],
			["yes", "maybe", "no"],
			["yes", "no"],
			["maybe", "no"],
			["no", "maybe"],
			["no", "yes"],
			["yes", "no"],
		]
	]

	openai_client = model.openai_client()
	start_step = await model.get_step()
	print(f"[script] Starting from step {start_step}")
	for step in range(start_step, start_step + 2):
		print(f"[script] Gather step={step}...")
		train_groups = await art.gather_trajectory_groups(
			(
				art.TrajectoryGroup(rollout(openai_client, prompt) for _ in range(8))
				for prompt in prompts
			),
			pbar_desc="gather",
		)
		print("[script] Training...")
		await model.train(
			train_groups,
			config=art.TrainConfig(learning_rate=1e-4),
			# _config=art.dev.TrainConfig(precalculate_logprobs=True),
		)
		print("[script] Train step complete")
	print("[script] Done.")


if __name__ == "__main__":
	# Allow long server spin-up
	os.environ.setdefault("ART_SERVER_TIMEOUT", "360.0")
	asyncio.run(main())
