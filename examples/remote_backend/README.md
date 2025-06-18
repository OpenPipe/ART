# Remote Backend Example

This example demonstrates how to connect to an ART server running on a
remote machine. The training loop is the same as in the local examples,
but all inference and training happen on the remote server.

1. Start the ART server on a machine with a GPU:

```bash
uv run art --host 0.0.0.0 --port 7999
```

2. On your local machine, set `ART_SERVER_URL` to the server's address and
run `remote_2048.py`:

```bash
ART_SERVER_URL=http://<server-ip>:7999 python remote_2048.py
```

The script will register the model with the remote server, gather
trajectories, and train the model remotely while you drive the loop
locally.
