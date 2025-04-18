## Contributing to ART

Clone the repository:

```bash
git clone https://github.com/OpenPipe/agent-reinforcement-training.git
cd agent-reinforcement-training
```

Install the dependencies:

```bash
uv sync
```

Then follow the SkyPilot or Local Training instructions below.

> **Warning:** There is currently a bug with tool use functionality. The issue appears to be that vLLM does not return all the token log probabilities for tool use. Further investigation is needed to determine the exact cause. For now, teaching use case-specific tool use with non-tool use models is the recommended workaround.

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
