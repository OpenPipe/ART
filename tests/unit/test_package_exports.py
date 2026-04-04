import art


def test_art_localbackend_top_level_export():
    from art.local import LocalBackend

    assert art.LocalBackend is LocalBackend
