import asyncio
import unsloth
from datasets import Dataset
import nest_asyncio
import os
import peft
import torch
import transformers
from trl import GRPOConfig, GRPOTrainer
from typing import cast
from typing import TYPE_CHECKING

from ..config.model import ModelConfig

if TYPE_CHECKING:
    import vllm
    from vllm.worker.worker_base import WorkerWrapperBase
    from vllm.worker.multi_step_model_runner import MultiStepModelRunner

    from .service import TuneInputs

nest_asyncio.apply()


class CausallLM(transformers.PreTrainedModel, transformers.GenerationMixin): ...


class ModelState:
    def __init__(self, config: ModelConfig) -> None:
        from vllm.engine import async_llm_engine

        # Set effectively unlimited timeout to support engine pausing & resumption
        async_llm_engine.ENGINE_ITERATION_TIMEOUT_S = 2**31 - 1
        # Sticking with V0 engine for now
        os.environ["VLLM_USE_V1"] = "0"
        # We can't use expandable segments with sleep mode
        enable_sleep_mode = config.get("init_args", {}).get("enable_sleep_mode", False)
        if enable_sleep_mode:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
        # Initialize Unsloth model
        self.model, self.tokenizer = cast(
            tuple[CausallLM, transformers.PreTrainedTokenizerBase],
            unsloth.FastLanguageModel.from_pretrained(**config.get("init_args", {})),
        )
        self.vllm = vLLMState(
            cast("vllm.AsyncLLMEngine", self.model.vllm_engine), enable_sleep_mode
        )
        # Initialize PEFT model
        self.peft_model = cast(
            peft.peft_model.PeftModelForCausalLM,
            unsloth.FastLanguageModel.get_peft_model(
                self.model, **config.get("peft_args", {})
            ),
        )
        self.lora_model = cast(peft.tuners.lora.LoraModel, self.peft_model.base_model)
        # Initialize trainer
        self.trainer = GRPOTrainer(
            model=self.peft_model,  # type: ignore
            reward_funcs=[],
            args=GRPOConfig(**config.get("train_args", {})),
            train_dataset=Dataset.from_list([{"prompt": ""} for _ in range(100_000)]),
            processing_class=self.tokenizer,
        )
        self.inputs_queue = asyncio.Queue["TuneInputs"]()

        # Patch trainer _prepare_inputs()
        def _async_prepare_inputs(*_, **__) -> dict[str, torch.Tensor]:
            async def get_inputs() -> "TuneInputs":
                return await self.inputs_queue.get()

            # Force otherwise synchronous _prepare_inputs() to yield
            # with nested asyncio.run() call
            inputs = asyncio.run(get_inputs())

            return cast(dict[str, torch.Tensor], inputs)

        self.trainer._prepare_inputs = _async_prepare_inputs


class vLLMState:
    def __init__(
        self, async_engine: "vllm.AsyncLLMEngine", enable_sleep_mode: bool
    ) -> None:
        from .vllm import create_engine_pause_and_resume_functions, patch_allocator

        if enable_sleep_mode:
            patch_allocator()
        self.async_engine = async_engine
        self.pause_engine, self.resume_engine = (
            create_engine_pause_and_resume_functions(self.async_engine)
        )
        self.driver_worker = cast(
            "WorkerWrapperBase",
            getattr(self.async_engine.engine.model_executor, "driver_worker"),
        )
        self.multi_step_model_runner: "MultiStepModelRunner" = (
            self.driver_worker.model_runner
        )
