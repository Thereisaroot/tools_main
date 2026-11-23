import subprocess
import time

import pytest

from shooklink.live_update.manager import State, SWITCH_EXIT
from shooklink.live_update import supervisor
from shooklink.live_update.supervisor import supervise


def test_failed_candidate_exits_before_last_good_is_relaunched(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    job = "1" * 32
    slot = tmp_path / "slots" / job
    slot.mkdir(parents=True)
    launches = []
    alive = []

    class Child:
        def __init__(self, folder):
            self.folder = folder
            self.code = None

        def wait(self, timeout=None):
            if len(launches) == 1:
                State(job=job, phase="restarting", target="a" * 40, highest=1).save(tmp_path)
                self.code = SWITCH_EXIT
            elif self.code is None:
                self.code = 0
            alive.clear()
            return self.code

        def poll(self):
            return self.code

        def terminate(self):
            self.code = 1
            alive.clear()

    def launch(folder):
        assert not alive, "two serial owners started concurrently"
        launches.append(folder)
        child = Child(folder)
        alive.append(child)
        return child

    assert supervise(tmp_path, bootstrap, launch, trial_timeout=0.02, poll=0.005) == 0
    assert launches == [bootstrap, slot, bootstrap]
    state = State.load(tmp_path)
    assert state.phase == "rolled_back" and state.active == "" and state.job == job


def test_supervisor_restart_recovers_interrupted_switch(tmp_path):
    active = "2" * 32
    State(active=active, phase="trial", job="3" * 32, target="a" * 40, highest=4).save(tmp_path)
    launches = []

    class Child:
        def wait(self, timeout=None):
            return 0
        def poll(self):
            return 0

    def launch(folder):
        launches.append(folder)
        return Child()

    assert supervise(tmp_path, tmp_path / "bootstrap", launch) == 0
    assert launches == [tmp_path / "slots" / active]
    assert State.load(tmp_path).phase == "rolled_back"


def test_stuck_old_gui_is_reaped_before_restart_and_candidate_is_not_started(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    launches = []
    class Child:
        code = None
        def poll(self):
            return self.code
        def wait(self, timeout=None):
            if len(launches) == 1 and self.code is None:
                State(phase="restarting", job="f" * 32).save(tmp_path)
                time.sleep(0.005)
                raise subprocess.TimeoutExpired("test", timeout)
            self.code = 0
            return self.code
        def terminate(self):
            self.code = 1

    def launch(folder):
        launches.append(folder)
        return Child()

    assert supervise(tmp_path, bootstrap, launch, handoff_timeout=0.01) == 0
    assert launches == [bootstrap, bootstrap]
    assert State.load(tmp_path).phase == "rolled_back"


def test_publisher_requires_local_enrollment_and_is_pinned(tmp_path, monkeypatch):
    from shooklink.local_control import client

    monkeypatch.setattr(client, "application_data_dir", lambda: tmp_path)
    monkeypatch.setattr(supervisor, "supervise", lambda *_: 0)
    assert supervisor.run_launcher(None, False) == 0
    assert not (tmp_path / "managed-updates" / "publisher").exists()
    assert supervisor.run_launcher("a" * 64, False) == 0
    assert supervisor.run_launcher(None, False) == 0
    with pytest.raises(ValueError, match="another publisher"):
        supervisor.run_launcher("b" * 64, False)
