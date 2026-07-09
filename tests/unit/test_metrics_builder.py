import asyncio

import pytest

from art.metrics import MetricsBuilder


class TestMetricsBuilder:
    @pytest.mark.asyncio
    async def test_rollup_correctness_across_depths(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        builder.add_cost("train/llm_judge/general_judge", usd=0.08)
        builder.add_cost("train/llm_judge/hallucination_judge", usd=0.04)
        builder.add_cost("train/tinker_train", usd=1.20)
        builder.add_cost("train/tinker_inference", usd=0.45)
        builder.add_cost("eval/llm_judge/correctness", usd=0.06)

        metrics = await builder.flush()

        assert metrics["costs/train/llm_judge"] == pytest.approx(0.12)
        assert metrics["costs/train"] == pytest.approx(1.77)
        assert metrics["costs/eval"] == pytest.approx(0.06)
        assert metrics["costs/all"] == pytest.approx(1.83)
        assert metrics["costs/cum/train/llm_judge"] == pytest.approx(0.12)
        assert metrics["costs/cum/train"] == pytest.approx(1.77)
        assert metrics["costs/cum/all"] == pytest.approx(1.83)

    @pytest.mark.asyncio
    async def test_cum_accumulates_for_hierarchical_sections(self) -> None:
        builder = MetricsBuilder(cost_context="train")

        builder.add_user_timing(step_wall_s=1.5, step_rollout_s=0.3)
        builder.add_data(
            step_num_scenarios=2,
            step_rollout_tokens=10,
            scenario_ids=["a", "b"],
        )
        first = await builder.flush()

        assert first["time/cum/wall_s"] == pytest.approx(1.5)
        assert first["time/cum/rollout_s"] == pytest.approx(0.3)
        assert first["data/cum/num_scenarios"] == pytest.approx(2)
        assert first["data/cum/rollout_tokens"] == pytest.approx(10)
        assert first["data/cum/num_unique_scenarios"] == 2

        builder.add_user_timing(step_wall_s=0.5, step_rollout_s=0.2)
        builder.add_data(
            step_num_scenarios=3,
            step_rollout_tokens=5,
            scenario_ids=["b", "c"],
        )
        second = await builder.flush()

        assert second["time/cum/wall_s"] == pytest.approx(2.0)
        assert second["time/cum/rollout_s"] == pytest.approx(0.5)
        assert second["data/cum/num_scenarios"] == pytest.approx(5)
        assert second["data/cum/rollout_tokens"] == pytest.approx(15)
        assert second["data/cum/num_unique_scenarios"] == 3

    @pytest.mark.asyncio
    async def test_helper_metrics_accumulate_within_a_single_step(self) -> None:
        builder = MetricsBuilder(cost_context="train")

        builder.add_data(step_num_scenarios=2, step_rollout_tokens=10)
        builder.add_data(step_num_scenarios=3, step_rollout_tokens=5)
        builder.add_user_timing(step_wall_s=1.5, step_rollout_s=0.3, step_eval_s=0.2)
        builder.add_user_timing(step_wall_s=0.5, step_rollout_s=0.2, step_eval_s=0.1)
        builder.add_idle_times(step_trainer_idle_s=1.0, step_rollout_idle_s=2.0)
        builder.add_idle_times(step_trainer_idle_s=0.5, step_rollout_idle_s=1.0)

        metrics = await builder.flush()

        assert metrics["data/step_num_scenarios"] == pytest.approx(5)
        assert metrics["data/step_rollout_tokens"] == pytest.approx(15)
        assert metrics["time/step_wall_s"] == pytest.approx(2.0)
        assert metrics["time/step_rollout_s"] == pytest.approx(0.5)
        assert metrics["time/step_eval_s"] == pytest.approx(0.3)
        assert metrics["time/step_trainer_idle_s"] == pytest.approx(1.5)
        assert metrics["time/step_rollout_idle_s"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_throughput_metrics_derive_from_time_and_token_cumulatives(
        self,
    ) -> None:
        builder = MetricsBuilder(cost_context="train")

        builder.add_metric("time/step_backend_train_s", 4.0)
        builder.add_metric("data/step_trainable_assistant_tokens", 40.0)
        builder.add_metric("time/step_rollout_s", 2.0)
        builder.add_metric("data/step_rollout_tokens", 10.0)
        builder.add_idle_times(step_trainer_idle_s=1.5, step_rollout_idle_s=0.5)

        metrics = await builder.flush()

        assert metrics["time/cum/trainer_idle_s"] == pytest.approx(1.5)
        assert metrics["time/cum/rollout_idle_s"] == pytest.approx(0.5)
        assert metrics["throughput/avg_trainer_tok_per_s"] == pytest.approx(10.0)
        assert metrics["throughput/avg_rollout_tok_per_s"] == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_costs_all_generated_for_single_and_multiple_children(self) -> None:
        single = MetricsBuilder(cost_context="train")
        single.add_cost("train/gpu", usd=2.0)
        one = await single.flush()
        assert one["costs/all"] == pytest.approx(2.0)

        multi = MetricsBuilder(cost_context="train")
        multi.add_cost("train/gpu", usd=2.0)
        multi.add_cost("eval/llm_judge/correctness", usd=0.5)
        two = await multi.flush()
        assert two["costs/all"] == pytest.approx(2.5)

    def test_leaf_parent_conflicts_raise(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        builder.add_cost("train", usd=1.0)
        with pytest.raises(ValueError):
            builder.add_cost("train/llm_judge", usd=0.1)

        other = MetricsBuilder(cost_context="train")
        other.add_cost("train/llm_judge", usd=0.1)
        with pytest.raises(ValueError):
            other.add_cost("train", usd=1.0)

    @pytest.mark.asyncio
    async def test_duplicate_leaf_writes_are_summed(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        builder.add_cost("train/gpu", usd=1.25)
        builder.add_cost("train/gpu", usd=0.75)

        metrics = await builder.flush()

        assert metrics["costs/train/gpu"] == pytest.approx(2.0)
        assert metrics["costs/train"] == pytest.approx(2.0)
        assert metrics["costs/all"] == pytest.approx(2.0)

    def test_cumulative_namespace_is_reserved(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        with pytest.raises(ValueError):
            builder.add_metric("costs/cum/train/llm_judge", 0.1)

    @pytest.mark.asyncio
    async def test_sparse_steps_omit_rollup_for_missing_costs(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        builder.add_cost("train/gpu", usd=1.0)
        first = await builder.flush()
        assert first["costs/cum/train"] == pytest.approx(1.0)

        second = await builder.flush()
        assert not any(key.startswith("costs/") for key in second)

        builder.add_cost("train/gpu", usd=2.0)
        third = await builder.flush()
        assert third["costs/train"] == pytest.approx(2.0)
        assert third["costs/cum/train"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_state_dict_round_trip_preserves_cumulative_state(self) -> None:
        before = MetricsBuilder(cost_context="train")
        before.add_cost("train/gpu", usd=1.0)
        await before.flush()

        state = before.state_dict()
        after = MetricsBuilder(cost_context="train")
        after.load_state_dict(state)
        after.add_cost("train/gpu", usd=2.0)

        metrics = await after.flush()
        assert metrics["costs/cum/train"] == pytest.approx(3.0)
        assert metrics["costs/cum/all"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_loaded_state_is_shared_with_other_cost_contexts(self) -> None:
        before = MetricsBuilder(cost_context="train")
        before.add_cost("train/gpu", usd=1.0)
        await before.flush()

        after = MetricsBuilder(cost_context="train")
        after.load_state_dict(before.state_dict())

        eval_builder = after.for_cost_context("eval")
        eval_builder.add_cost("eval/judge", usd=2.0)

        metrics = await eval_builder.flush()
        assert metrics["costs/eval/judge"] == pytest.approx(2.0)
        assert metrics["costs/cum/all"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_add_response_cost_uses_registered_model_pricing(self) -> None:
        builder = MetricsBuilder(cost_context="eval")
        builder.register_model_pricing(
            "anthropic/test-judge",
            prompt_per_million=5.0,
            completion_per_million=7.0,
        )

        cost = builder.add_response_cost(
            "llm_judge/faithfulness",
            {
                "model": "anthropic/test-judge",
                "usage": {"input_tokens": 40, "output_tokens": 60},
            },
            provider="anthropic",
            model_name="anthropic/test-judge",
        )

        metrics = await builder.flush()
        assert cost == pytest.approx(0.00062)
        assert metrics["costs/eval/llm_judge/faithfulness"] == pytest.approx(0.00062)

    @pytest.mark.asyncio
    async def test_unique_scenario_count_tracks_exact_ids(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        builder.add_data(scenario_ids=["s1", "s2", "s3"])
        first = await builder.flush()
        assert first["data/cum/num_unique_scenarios"] == 3

        builder.add_data(scenario_ids=["s2", "s4"])
        second = await builder.flush()
        assert second["data/cum/num_unique_scenarios"] == 4

    @pytest.mark.asyncio
    async def test_empty_flush_does_not_repeat_stale_derived_metrics(self) -> None:
        builder = MetricsBuilder(cost_context="train")
        builder.add_metric("time/step_backend_train_s", 2.0)
        builder.add_metric("data/step_trainable_assistant_tokens", 20.0)
        builder.add_data(scenario_ids=["s1"])

        first = await builder.flush()
        assert first["throughput/avg_trainer_tok_per_s"] == pytest.approx(10.0)
        assert first["data/cum/num_unique_scenarios"] == 1

        second = await builder.flush()
        assert second == {}

    @pytest.mark.asyncio
    async def test_concurrent_add_cost_calls_do_not_lose_updates(self) -> None:
        builder = MetricsBuilder(cost_context="train")

        async def worker() -> None:
            for _ in range(25):
                builder.add_cost("train/gpu", usd=0.1)
                await asyncio.sleep(0)

        await asyncio.gather(*(worker() for _ in range(4)))
        metrics = await builder.flush()

        assert metrics["costs/train/gpu"] == pytest.approx(10.0)
        assert metrics["costs/all"] == pytest.approx(10.0)

    def test_contextvar_activate_and_get_active(self) -> None:
        builder = MetricsBuilder(cost_context="eval")
        token = builder.activate()
        assert MetricsBuilder.get_active() is builder
        token.var.reset(token)
