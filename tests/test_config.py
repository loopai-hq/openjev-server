"""Profile settings: a CLI flag wins over the released helper's READOUT_* variables, which win over the profile file."""

import pytest

from openjev_server import cli
from openjev_server.config import HELPER_ENV, load_profile


@pytest.fixture(autouse=True)
def no_helper_env(monkeypatch):
    """The compat tests export the released helper's READOUT_* settings for the whole session; these tests start without them."""
    for env in HELPER_ENV.values():
        monkeypatch.delenv(env, raising=False)


def test_read_twice_is_off_by_default():
    assert load_profile("openjev").repeat is False and load_profile("uncalibrated").repeat is False


@pytest.mark.parametrize(("value", "on"), [("1", True), ("0", False), ("true", False)])
def test_readout_repeat_reads_as_in_the_helper(monkeypatch, value, on):
    """READOUT_REPEAT is on only for "1", the released helper's rule; the CLI flag wins over it."""
    monkeypatch.setenv("READOUT_REPEAT", value)
    assert load_profile("openjev").repeat is on
    assert load_profile("openjev", {"repeat": True}).repeat is True


def test_repeat_flag_reaches_the_profile(monkeypatch):
    seen = {}

    def stop(name, overrides):
        seen.update(overrides)
        raise SystemExit

    monkeypatch.setattr(cli, "load_profile", stop)
    with pytest.raises(SystemExit):
        cli.main(["serve", "--repeat"])
    assert seen["repeat"] is True
