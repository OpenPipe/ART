# ART-E Email Search Example

This example is a lightweight, local version of the ART-E email search task. It
shows how to train an agent to search a small inbox, read relevant messages, and
return a grounded answer with supporting message IDs.

The example is intentionally small:

- No external email service is required.
- The inboxes are deterministic Python fixtures.
- The rollout uses a simple text protocol instead of provider-specific tool
  calling so it works across most chat models.
- The reward combines exact answer matching and citation correctness.

For the full ART-E research context, see the
[ART-E blog post](https://openpipe.ai/blog/art-e-mail-agent).

## Files

- `scenarios.py` defines inbox fixtures, search/read helpers, and answer
  scoring.
- `rollout.py` runs one multi-turn email-search trajectory.
- `train.py` trains a small model with ART using the local scenarios.

## Run One Rollout

Set an inference API key for the provider used by your `art.Model`, then run:

```bash
python examples/art_e/rollout.py
```

The script uses an OpenRouter model by default for a cheap smoke test. You can
change the model configuration at the bottom of `rollout.py`.

## Train

Training requires the normal ART local backend setup:

```bash
python examples/art_e/train.py
```

The default training configuration is deliberately modest so the example is easy
to inspect. Increase `SIMULTANEOUS_ROLLOUTS`, `TRAIN_STEPS`, or the base model
when running on a larger GPU.
