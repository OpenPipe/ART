## Contributing to ART

Clone the repository:

```bash
git clone https://github.com/OpenPipe/ART.git
cd ART
```

Install the dependencies:

```bash
uv sync
```

### Code Formatting and Linting

This project uses [ruff](https://github.com/astral-sh/ruff) for both code formatting and linting. Before submitting a pull request, please ensure your code passes all quality checks:

```bash
# Run all code quality checks (formatting, linting, and dependency sync)
./scripts/run_checks.sh

# Automatically fix any issues that can be fixed
./scripts/run_checks.sh --fix
```

The `run_checks.sh` script will:

1. Check code formatting with ruff
2. Check for linting issues with ruff
3. Verify that `uv.lock` is in sync with `pyproject.toml`

These checks are automatically run in CI for all pull requests. If your PR fails these checks, simply run `./scripts/run_checks.sh --fix` locally and commit the changes.

### Release Process

To create a new release:

1. **Review merged PRs since the last release**:
   - Go to the [pull requests page](https://github.com/OpenPipe/ART/pulls?q=is%3Apr+is%3Amerged+sort%3Aupdated-desc)
   - Review PRs merged since the last release to understand what changed
   - Note any breaking changes, new features, or important bug fixes

2. **Create a draft release**:
   - Go to [Actions](https://github.com/OpenPipe/ART/actions/workflows/create-draft-release.yml)
   - Click "Run workflow"
   - Select the version bump type:
     - `patch`: Bug fixes and minor changes (0.3.13 → 0.3.14)
     - `minor`: New features and non-breaking changes (0.3.13 → 0.4.0)  
     - `major`: Breaking changes (0.3.13 → 1.0.0)

3. **Edit the draft release notes**:
   - Go to the [releases page](https://github.com/OpenPipe/ART/releases)
   - Click "Edit" on the draft release
   - Add release highlights, breaking changes, and curated changelog
   - The auto-generated PR list provides a starting point, but manual curation improves clarity

4. **Finalize the release**:
   - Review and merge the automatically created release PR
   - This will automatically:
     - Create the git tag
     - Publish the curated release notes
     - Build and publish the package to PyPI

Then follow the SkyPilot or Local Training instructions below.

### SkyPilot

Copy the `.env.example` file to `.env` and set the environment variables:

```bash
cp .env.example .env
```

Ensure you have a valid SkyPilot cloud available:

```bash
uv run sky check
```

Launch a cluster:

```bash
./scripts/launch-cluster.sh # you can pass any sky launch arguments here
```

Make sure you are on a machine with at least one H100 or A100-80GB GPU. Machines equipped with lower-end GPUs may work, but training will be slower.

You can now SSH into the `art` cluster, using either VSCode or the command line.

### Connecting via Command Line

Simply run:

```bash
ssh art
```

### Connecting via VSCode

1. **Install the Remote-SSH extension on your local machine**

   - Open the extensions view by clicking on the Extensions icon in the Activity Bar on the left.
   - Search for **"Remote-SSH"** and install it.

2. **Configure default extensions for your remote host**

   - In your VSCode settings, find **"Remote.SSH: Default Extensions"**
   - Add the following extensions:
     - `ms-python.python`
     - `ms-toolsai.jupyter`
     - `eamodio.gitlens`
     - `charliermarsh.ruff`

3. **Connect to the host**

   - Open the command palette and run **"Remote-SSH: Connect to Host..."**
   - Select `art`

4. **Set up the host**

   - Click **"Open Folder"**
     - Select **"sky_workdir"**
     - Click **OK**

5. **Run a notebook**
   - Find `2048.ipynb` and run it!

### "2048" example

Now you can run the "2048" example in `/examples/2048/2048.ipynb`.

It has been tested with the `Qwen/Qwen2.5-14B-Instruct` model on a 1xH100 instance.

You can monitor training progress with Weights & Biases at https://wandb.ai/your-wandb-organization/agent-reinforcement-training.

You should see immediate improvement in `val/reward` after one step.

If you run into any issues, the training output is set to maximum verbosity. Copying the outputs such as the vLLM or torchtune logs, or copying/screenshotting the plotted packed tensors, may help me debug the issue.

### Cleaning Up

When you're done, you can tear down the cluster with:

```bash
uv run sky down art
```
