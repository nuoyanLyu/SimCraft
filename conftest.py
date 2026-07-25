import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: hits a real external API (costs money/quota); excluded by default, run with `-m live`",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("-m") == "live":
        return
    skip_live = pytest.mark.skip(reason="live test: run explicitly with `pytest -m live`")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
