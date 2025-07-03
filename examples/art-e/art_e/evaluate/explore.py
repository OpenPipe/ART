# %%


from art_e.evaluate.load_trajectories import load_trajectories

# await load_trajectories.bust_cache()
df = await load_trajectories(
    ".art/email_agent",
    models=["email-agent-008"],
)  # type: ignore
