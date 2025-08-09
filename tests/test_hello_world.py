import art


def test_can_import_art_and_create_trajectory():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello world."},
        {"role": "assistant", "content": "Hello, world!"},
    ]

    traj = art.Trajectory(messages_and_choices=messages, reward=1.0)
    # Basic sanity checks
    assert traj.reward == 1.0
    assert len(traj.messages()) == 3
    # Finish should add a duration metric
    traj.finish()
    assert "duration" in traj.metrics


