# Proposal: Backend-First Training API

## Summary

Replace the current `model.train(trajectory_groups, config)` API with a backend-first `backend.train(model, trajectory_groups, ...)` API. This provides stronger type safety, eliminates the need for a generic config object, and allows each backend to define its own semantically meaningful parameters.

## Current Architecture

Training is currently invoked through the model:

```python
model = art.TrainableModel(name="my-model", project="my-project", base_model="...")
await model.register(backend)

# Training call
await model.train(
    trajectory_groups,
    config=TrainConfig(learning_rate=5e-6),
    _config=dev.TrainConfig(epsilon=0.2, ppo=True),  # experimental
)
```

### Problems with the Current Approach

#### 1. Weak Type Safety

The `dev.TrainConfig` is a `TypedDict` with ~17 optional fields, but **different backends support different subsets**:

| Argument | ServerlessBackend | LocalBackend | TinkerBackend |
|----------|:-----------------:|:------------:|:-------------:|
| `learning_rate` | ✅ | ✅ | ✅ |
| `advantage_balance` | ✅ | ✅ | ✅ |
| `epsilon` / `epsilon_high` | ✅ | ✅ | ✅ |
| `ppo` | ✅ | ✅ | ✅ |
| `allow_training_without_logprobs` | ❌ | ✅ | ✅ |
| `plot_tensors` | ❌ | ✅ | ✅ |
| `truncated_importance_sampling` | ❌ | ✅ | ✅ |
| `logprob_calculation_chunk_size` | ❌ | ✅ | ❌ |

Currently, users can pass `plot_tensors=True` when using `ServerlessBackend` and it will be silently ignored. There's no way for type checkers or IDEs to warn about this.

#### 2. Config Object Obscures Intent

The two-config pattern (`TrainConfig` + `dev.TrainConfig`) exists because some args are "stable" and some are "experimental". But this doesn't map to the real axis of variation, which is **which backend you're using**.

#### 3. Backend-Specific Behaviors Don't Fit the Model

Different backends have fundamentally different capabilities:

- **TinkerBackend**: Can save checkpoints to Tinker's cloud storage, optionally skip saving for faster iteration, control which layers to train
- **LocalBackend**: Saves checkpoints to disk, supports full HuggingFace `TrainerArgs`, can use Torchtune for multi-GPU
- **ServerlessBackend**: Checkpoints are W&B artifacts, no local storage, different checkpoint lifecycle

These don't fit naturally into "pass config to model.train()". For example, TinkerBackend might want:

```python
# This makes no sense on LocalBackend
await model.train(..., save_checkpoint=False, deploy_to_inference=True)
```

#### 4. Model Doesn't Own Training

The model is a *specification* (name, project, base_model, config). The backend *owns* the training infrastructure. Having `model.train()` delegate to `backend._train_model()` inverts the natural ownership.

---

## Proposed Architecture

### New API

```python
model = art.TrainableModel(name="my-model", project="my-project", base_model="...")
await model.register(backend)

# Backend-first training call
await backend.train(
    model,
    trajectory_groups,
    learning_rate=5e-6,
    # Backend-specific args are type-checked
)
```

### Backend-Specific Signatures

Each backend defines its own `train()` method with appropriate parameters:

#### ServerlessBackend

```python
class ServerlessBackend(Backend):
    async def train(
        self,
        model: TrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        # Core training args
        learning_rate: float = 5e-6,
        
        # RL algorithm settings
        ppo: bool = False,
        epsilon: float | None = None,  # defaults based on ppo
        epsilon_high: float | None = None,
        
        # Advantage computation
        advantage_balance: float = 0.0,
        scale_rewards: bool = True,
        
        # Importance sampling
        importance_sampling_level: Literal["token", "sequence", "average", "geometric_average"] = "token",
        max_negative_advantage_importance_sampling_weight: float | None = None,
        mask_prob_ratio: bool = False,
        
        # Experimental
        kimi_k2_tau: float | None = None,
        precalculate_logprobs: bool = False,
    ) -> TrainResult:
        ...
```

#### LocalBackend

```python
class LocalBackend(Backend):
    async def train(
        self,
        model: TrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        # All ServerlessBackend args, plus:
        learning_rate: float = 5e-6,
        ppo: bool = False,
        epsilon: float | None = None,
        epsilon_high: float | None = None,
        advantage_balance: float = 0.0,
        scale_rewards: bool = True,
        importance_sampling_level: Literal["token", "sequence", "average", "geometric_average"] = "token",
        max_negative_advantage_importance_sampling_weight: float | None = None,
        mask_prob_ratio: bool = False,
        kimi_k2_tau: float | None = None,
        precalculate_logprobs: bool = False,
        
        # LocalBackend-specific
        allow_training_without_logprobs: bool = False,
        plot_tensors: bool = False,
        truncated_importance_sampling: float | None = None,
        scale_learning_rate_by_reward_std_dev: bool = False,
        logprob_calculation_chunk_size: int = 1024,
        
        # Checkpoint behavior
        save_checkpoint: bool = True,
        
        verbose: bool = False,
    ) -> TrainResult:
        ...
```

#### TinkerBackend

```python
class TinkerBackend(Backend):
    async def train(
        self,
        model: TrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        # Core args (subset that Tinker supports)
        learning_rate: float = 5e-6,
        ppo: bool = False,
        epsilon: float | None = None,
        epsilon_high: float | None = None,
        advantage_balance: float = 0.0,
        scale_rewards: bool = True,
        importance_sampling_level: Literal["token", "sequence", "average", "geometric_average"] = "token",
        
        # Tinker-specific checkpoint behavior
        save_checkpoint: bool = True,
        deploy_checkpoint: bool = False,  # Push to Tinker inference
        
        # Tinker-specific training options
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = False,
        
        verbose: bool = False,
    ) -> TrainResult:
        ...
```

### Type Safety Benefits

With the backend-first API, type checkers can validate arguments:

```python
# ✅ Valid - LocalBackend supports plot_tensors
local_backend = LocalBackend()
await local_backend.train(model, groups, plot_tensors=True)

# ❌ Type error - ServerlessBackend doesn't have plot_tensors
serverless_backend = ServerlessBackend()
await serverless_backend.train(model, groups, plot_tensors=True)  # pyright/mypy error!

# ❌ Type error - TinkerBackend doesn't have logprob_calculation_chunk_size
tinker_backend = TinkerBackend()
await tinker_backend.train(model, groups, logprob_calculation_chunk_size=512)  # Error!
```

IDEs will provide accurate autocomplete showing only the arguments available for the specific backend being used.

### Backend-Specific Behaviors

The new API naturally accommodates backend-specific behaviors:

```python
# TinkerBackend: Train without saving (for rapid iteration)
await tinker_backend.train(model, groups, save_checkpoint=False)

# TinkerBackend: Train and immediately deploy
await tinker_backend.train(model, groups, deploy_checkpoint=True)

# LocalBackend: Visualize training tensors
await local_backend.train(model, groups, plot_tensors=True)

# ServerlessBackend: Just works, minimal options
await serverless_backend.train(model, groups, learning_rate=1e-5)
```

---

## Migration Path

### Phase 1: Add `backend.train()` and deprecate `model.train()`

Add the new method and immediately deprecate the old one:

```python
# Old way (deprecated, emits warning)
await model.train(trajectory_groups, config=TrainConfig(learning_rate=5e-6))

# New way
await backend.train(model, trajectory_groups, learning_rate=5e-6)
```

```python
# In model.py
async def train(self, ...):
    warnings.warn(
        "model.train() is deprecated. Use backend.train(model, ...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    ...
```

### Phase 2: Remove `model.train()` and config objects

In the next major version, remove:
- `model.train()`
- `TrainConfig` class
- `dev.TrainConfig` class

---

## Additional Design Considerations

### Return Type

Instead of returning `None`, `backend.train()` should return a structured result:

```python
@dataclass
class TrainResult:
    step: int
    metrics: dict[str, float]
    checkpoint_path: str | None  # LocalBackend
    artifact_name: str | None    # ServerlessBackend
    deployed: bool               # TinkerBackend
```

### Logging Behavior

Currently `model.train()` calls `model.log()` internally. With the backend-first API, logging is explicit and separated from training:

```python
# Log trajectories, train, then log the returned metrics
await model.log(groups)
result = await backend.train(model, groups, learning_rate=5e-6)
await model.log(metrics=result.metrics, step=result.step)
```

For convenience, `model.log()` gains optional parameters to support logging metrics alongside or instead of trajectories:

```python
async def log(
    self,
    trajectory_groups: Iterable[TrajectoryGroup] | None = None,
    *,
    split: str = "train",
    metrics: dict[str, float] | None = None,
    step: int | None = None,
) -> None:
    ...
```

This enables a 1-liner pattern using the walrus operator:

```python
# 1-liner: log trajectories and metrics together
await model.log(groups, metrics=(result := await backend.train(model, groups)).metrics, step=result.step)
```

Or the more readable multi-line version when you need the result for other purposes:

```python
await model.log(groups)
result = await backend.train(model, groups, learning_rate=5e-6)
await model.log(metrics=result.metrics, step=result.step)

# Use result.step, result.checkpoint_path, etc.
```

This separation gives users full control:
- Skip logging entirely for debugging/iteration
- Log to different splits
- Log additional custom metrics alongside training metrics

### Protocol/ABC for Common Interface

If users need to write backend-agnostic code, we can provide a protocol:

```python
class TrainableBackend(Protocol):
    async def train(
        self,
        model: TrainableModel,
        trajectory_groups: Iterable[TrajectoryGroup],
        *,
        learning_rate: float = ...,
        # Only the common subset
    ) -> TrainResult:
        ...
```

---

## Summary

| Aspect | Current (`model.train()`) | Proposed (`backend.train()`) |
|--------|---------------------------|------------------------------|
| Type safety | Weak (generic TypedDict) | Strong (backend-specific signatures) |
| IDE support | Limited autocomplete | Full autocomplete per backend |
| Backend-specific features | Awkward fit | Natural expression |
| Config objects | Two (`TrainConfig` + `dev.TrainConfig`) | None (explicit kwargs) |
| Ownership semantics | Model delegates to backend | Backend owns training |
| Silent failures | Args ignored if unsupported | Type errors at call site |

The backend-first API better reflects the reality that training behavior is fundamentally determined by the backend, not the model. It provides stronger type safety, better IDE support, and a more natural place to express backend-specific capabilities.
