"""The equal-budget sweep works end to end, without spending a GPU hour on it.

``docs/RUN_THE_SWEEP.md`` asks for about six hours of a free-tier T4 across
three sessions. The plumbing between those sessions -- tuning caches into the
state file, evaluation reuses it, seeds stay disjoint, a resumed run skips what
is done -- is exactly the kind of thing that fails on the second session, two
hours in, with the first session's runtime already reclaimed.

So the whole path runs here against a stubbed trainer: real argument parsing,
real state file, real tuning loop, real grid, fake losses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import astro_lab  # noqa: E402


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    """Run astro_lab.main() in a scratch directory with training stubbed out."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(astro_lab, "STATE", Path("astro_lab_state.json"))

    calls: list[dict] = []

    def fake_train(name, config, seed, **kwargs):
        calls.append({"name": name, "seed": seed, "config": dict(config),
                      "steps": kwargs["steps"]})
        # A deterministic surface with a per-optimizer optimum, so the tuner has
        # something real to find rather than noise.
        best = {"muon": 0.02, "normuon": 0.02, "astro": 0.01,
                "astro_trust": 0.003}.get(name, 0.02)
        penalty = abs(config["lr"] - best) / best
        # Report one second per step. The budget estimator reads this field to
        # predict the next run's cost, so a stub that under-reports its own
        # duration would let a broken estimator pass.
        return 6.5 + 0.1 * penalty + 0.001 * seed, float(kwargs["steps"])

    monkeypatch.setattr(astro_lab, "train_once", fake_train)
    monkeypatch.setattr(astro_lab, "load_tokens",
                        lambda tokenizer, needed, cache: __import__("torch").zeros(
                            needed + 10, dtype=__import__("torch").long))

    class Tokenizer:
        def __len__(self):
            return 50257

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: Tokenizer()))
    return calls


def run(argv: list[str]) -> int:
    sys.argv = ["astro_lab.py", *argv]
    return astro_lab.main()


BASE = ["--mode", "scaling", "--sizes", "124M", "--optimizers",
        "muon", "astro", "astro_trust"]


def test_tuning_gives_every_optimizer_the_same_number_of_trials(sweep) -> None:
    assert run([*BASE, "--steps", "300", "--trials", "4", "--seeds", "0"]) == 0
    tuning = [c for c in sweep if c["seed"] == 0]
    per_optimizer = {name: sum(1 for c in tuning if c["name"] == name)
                     for name in ("muon", "astro", "astro_trust")}
    assert set(per_optimizer.values()) == {4}, per_optimizer


def test_each_optimizer_is_tuned_in_its_own_range(sweep) -> None:
    """astro_trust measures its step as a fraction of a layer's norm, so a
    Muon-scaled range would be a silent handicap -- the bug this project has
    shipped twice."""
    run([*BASE, "--steps", "300", "--trials", "5", "--seeds", "0"])
    for name in ("muon", "astro", "astro_trust"):
        low, high = astro_lab.space_for(name)["lr"]
        drawn = [c["config"]["lr"] for c in sweep if c["name"] == name]
        assert drawn, name
        assert all(low <= value <= high for value in drawn), (name, drawn)
    muon_rates = [c["config"]["lr"] for c in sweep if c["name"] == "muon"]
    trust_rates = [c["config"]["lr"] for c in sweep if c["name"] == "astro_trust"]
    assert max(trust_rates) < max(muon_rates)


def test_tuning_seed_is_disjoint_from_evaluation_seeds(sweep) -> None:
    run([*BASE, "--steps", "300", "--trials", "3", "--seeds", "2"])
    tuning = {c["seed"] for c in sweep if c["config"] and c["seed"] == 0}
    evaluation = {c["seed"] for c in sweep if c["seed"] != 0}
    assert tuning == {0}
    assert evaluation == {100, 101}
    assert not (tuning & evaluation)


def test_a_second_session_reuses_the_first_session_tuning(sweep) -> None:
    """Sessions 2 and 3 of the runbook depend on this; if it fails the user
    burns two hours re-tuning."""
    run([*BASE, "--steps", "300", "--trials", "4", "--seeds", "0"])
    tuned = json.loads(Path("astro_lab_state.json").read_text())["tuned"]
    assert set(tuned) == {"muon", "astro", "astro_trust"}

    sweep.clear()
    run([*BASE, "--steps", "300", "--trials", "4", "--seeds", "2"])
    assert not [c for c in sweep if c["seed"] == 0], "re-tuned instead of reusing"
    assert len(sweep) == 3 * 2

    after = json.loads(Path("astro_lab_state.json").read_text())["tuned"]
    assert after == tuned


def test_a_resumed_run_skips_completed_cells(sweep) -> None:
    run([*BASE, "--steps", "300", "--trials", "2", "--seeds", "2"])
    done = len(sweep)
    sweep.clear()
    run([*BASE, "--steps", "300", "--trials", "2", "--seeds", "2"])
    assert sweep == [], f"repeated {len(sweep)} of {done} finished runs"


def test_a_longer_horizon_reuses_the_configuration_tuned_at_the_short_one(sweep) -> None:
    run([*BASE, "--steps", "300", "900", "--trials", "3", "--seeds", "1"])
    long_runs = [c for c in sweep if c["steps"] == 900 and c["seed"] != 0]
    short_runs = [c for c in sweep if c["steps"] == 300 and c["seed"] != 0]
    assert long_runs and short_runs
    for name in ("muon", "astro", "astro_trust"):
        short = next(c["config"] for c in short_runs if c["name"] == name)
        long = next(c["config"] for c in long_runs if c["name"] == name)
        assert short == long


def test_report_only_needs_no_gpu_and_no_corpus(sweep, monkeypatch) -> None:
    run([*BASE, "--steps", "300", "--trials", "2", "--seeds", "2"])

    def explode(*args, **kwargs):
        raise AssertionError("--report-only must not download or train")

    monkeypatch.setattr(astro_lab, "load_tokens", explode)
    monkeypatch.setattr(astro_lab, "train_once", explode)
    assert run([*BASE, "--steps", "300", "--report-only"]) == 0
    assert Path("astro_lab_report.md").is_file()
    assert "muon" in Path("astro_lab_report.md").read_text()


def test_refusing_to_run_untuned_is_explicit(sweep) -> None:
    with pytest.raises(SystemExit, match="no configuration"):
        run([*BASE, "--steps", "300", "--trials", "0", "--seeds", "2"])


# ---------------------------------------------------------------------------
# The time budget is a promise, not a suggestion
# ---------------------------------------------------------------------------


def test_a_run_that_cannot_finish_in_budget_is_not_started(sweep, monkeypatch) -> None:
    """The failure the caller least expects.

    ``--max-minutes 240`` was only checked *between* runs, so a 46-minute
    2700-step cell starting at minute 239 finished at 285. A real session ran
    four hours under a four-hour budget and returned nothing. The budget must
    account for how long the next run will take.
    """
    import time as time_module

    clock = {"now": 0.0}
    monkeypatch.setattr(astro_lab.time, "perf_counter", lambda: clock["now"])

    started = []
    real_train = astro_lab.train_once

    def slow_train(name, config, seed, **kwargs):
        started.append((name, seed, kwargs["steps"]))
        # Each 900-step run costs 15 minutes of the fake clock.
        clock["now"] += kwargs["steps"] * 1.0
        return real_train(name, config, seed, **kwargs)

    monkeypatch.setattr(astro_lab, "train_once", slow_train)
    assert time_module is not None  # the module is patched, not replaced

    # Budget of 40 minutes; each 900-step run needs 15 (+10% headroom = 16.5).
    run([*BASE, "--steps", "900", "--trials", "0", "--seeds", "3",
         "--config", "lr=0.0144", "--max-minutes", "40"])

    assert started, "nothing ran at all"
    assert len(started) <= 2, (
        f"started {len(started)} runs of ~16.5 min each inside a 40-minute "
        "budget; the last one would overrun"
    )


def test_the_budget_still_writes_a_report_when_it_stops_early(sweep, monkeypatch) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr(astro_lab.time, "perf_counter", lambda: clock["now"])
    real_train = astro_lab.train_once

    def slow_train(name, config, seed, **kwargs):
        clock["now"] += kwargs["steps"] * 1.0
        return real_train(name, config, seed, **kwargs)

    monkeypatch.setattr(astro_lab, "train_once", slow_train)
    run([*BASE, "--steps", "900", "--trials", "0", "--seeds", "3",
         "--config", "lr=0.0144", "--max-minutes", "40"])
    assert Path("astro_lab_report.md").is_file()


def test_the_cost_estimate_prefers_what_this_state_file_measured(sweep) -> None:
    """A fallback estimate is for the first run only; after that the script has
    real timings and should use them."""
    run([*BASE, "--steps", "300", "--trials", "0", "--seeds", "1",
         "--config", "lr=0.0144"])
    state = json.loads(Path("astro_lab_state.json").read_text())
    assert all(entry["seconds"] > 0 for entry in state["runs"].values())


def test_an_unknown_optimizer_name_is_refused_not_silently_substituted(sweep) -> None:
    """A placeholder from a document was passed literally, tuned, and would
    have been evaluated for hours as a default ASTRO under a name that was
    never an optimizer. Silent fallbacks in a comparison harness are how a
    table ends up describing something nobody ran."""
    with pytest.raises(SystemExit, match="unknown optimizer 'WINNER'"):
        run(["--mode", "scaling", "--sizes", "124M", "--steps", "300",
             "--optimizers", "muon", "WINNER", "--trials", "2", "--seeds", "1",
             "--config", "lr=0.0144"])


def test_the_refusal_lists_what_is_available(sweep) -> None:
    import astro_lab as lab

    with pytest.raises(SystemExit) as caught:
        lab.space_for("astro_v3")
    message = str(caught.value)
    assert "astro_v2" in message and "muon" in message


def test_every_advertised_name_actually_builds(sweep) -> None:
    """known_optimizers() is what the error message promises, so each entry
    has to be constructible."""
    import astro_lab as lab
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_layer=2, n_head=2, n_embd=64, n_positions=32,
                        vocab_size=128)
    for name in lab.known_optimizers():
        torch.manual_seed(0)
        model = GPT2LMHeadModel(config)
        low, high = lab.space_for(name)["lr"]
        draw = {"lr": (low * high) ** 0.5, "weight_decay": 0.01,
                "scalar_lr_mult": 0.1, "beta2": 0.95}
        assert lab.build_optimizer(name, model, draw) is not None, name


def test_tuning_can_use_a_cheaper_budget_than_evaluation(sweep) -> None:
    """Tuning at the evaluation budget is what makes a sweep unaffordable:
    five trials for six optimizers at 900 steps is seven hours, the same at
    300 steps is two."""
    run([*BASE, "--steps", "900", "--tune-steps", "300",
         "--trials", "3", "--seeds", "1"])
    tuning = [c for c in sweep if c["seed"] == 0]
    evaluation = [c for c in sweep if c["seed"] != 0]
    assert tuning and evaluation
    assert {c["steps"] for c in tuning} == {300}, "tuned at the wrong budget"
    assert {c["steps"] for c in evaluation} == {900}, "evaluated at the wrong budget"


def test_without_tune_steps_tuning_uses_the_first_grid_cell(sweep) -> None:
    run([*BASE, "--steps", "900", "--trials", "3", "--seeds", "1"])
    assert {c["steps"] for c in sweep if c["seed"] == 0} == {900}


def test_the_scalar_multiplier_range_admits_the_selected_value(sweep) -> None:
    """Two optimizers selected 0.4369 from (0.02, 0.5) -- the 96% point in log
    space. A range whose top is the answer has not been searched, it has been
    hit; the range must extend well past what was chosen."""
    import math

    low, high = astro_lab.space_for("muon")["scalar_lr_mult"]
    selected = 0.4368648366496585
    position = ((math.log(selected) - math.log(low))
                / (math.log(high) - math.log(low)))
    assert position < 0.8, (
        f"scalar_lr_mult={selected} sits at the {position:.0%} point of "
        f"({low}, {high}); widen the range"
    )


# ---------------------------------------------------------------------------
# Trial-level resume, pinning, and the paired-across-configurations readback
# ---------------------------------------------------------------------------


def test_an_interrupted_tuning_session_keeps_its_finished_trials(sweep) -> None:
    """A 165-minute session that ran out of budget at trial 7 of 8 used to
    discard six finished runs -- an hour and a half of a T4 -- because only the
    winner was written to the state file."""
    run([*BASE, "--steps", "300", "--trials", "3", "--seeds", "0"])
    trials = json.loads(Path("astro_lab_state.json").read_text())["trials"]
    assert sorted(trials) == sorted(
        f"124M|300|{name}|t{i}"
        for name in ("muon", "astro", "astro_trust") for i in range(3))
    for entry in trials.values():
        assert entry["value"] is not None
        assert entry["config"]["lr"] > 0


def test_a_resumed_session_does_not_re_run_finished_trials(sweep) -> None:
    run([*BASE, "--steps", "300", "--trials", "3", "--seeds", "0"])
    first = len(sweep)
    assert first == 9
    # Drop the tuned configurations but keep the trials, which is the state a
    # session interrupted mid-sweep leaves behind.
    state = json.loads(Path("astro_lab_state.json").read_text())
    state["tuned"] = {}
    Path("astro_lab_state.json").write_text(json.dumps(state))

    sweep.clear()
    run([*BASE, "--steps", "300", "--trials", "3", "--seeds", "0"])
    assert sweep == [], "recorded trials were re-run instead of read back"


def test_a_resumed_session_picks_the_same_winner_it_would_have(sweep) -> None:
    run([*BASE, "--steps", "300", "--trials", "4", "--seeds", "0"])
    complete = json.loads(Path("astro_lab_state.json").read_text())["tuned"]

    state = json.loads(Path("astro_lab_state.json").read_text())
    state["tuned"] = {}
    # Forget the last trial of each optimizer, as an interruption would.
    state["trials"] = {k: v for k, v in state["trials"].items()
                       if not k.endswith("|t3")}
    Path("astro_lab_state.json").write_text(json.dumps(state))
    sweep.clear()
    run([*BASE, "--steps", "300", "--trials", "4", "--seeds", "0"])
    resumed = json.loads(Path("astro_lab_state.json").read_text())["tuned"]

    assert set(resumed) == set(complete)
    for name in complete:
        assert resumed[name] == complete[name], name
    # Only the forgotten trial was re-run, not the whole sweep.
    assert len(sweep) == 3, [c["name"] for c in sweep]


def test_pinning_holds_a_hyperparameter_fixed_and_still_tunes_the_rest(sweep) -> None:
    run([*BASE, "--steps", "300", "--trials", "4", "--seeds", "0",
         "--pin", "scalar_lr_mult=0.4369"])
    for name in ("muon", "astro", "astro_trust"):
        drawn = [c["config"] for c in sweep if c["name"] == name]
        assert len(drawn) == 4, name
        assert {c["scalar_lr_mult"] for c in drawn} == {0.4369}, name
        assert len({c["lr"] for c in drawn}) == 4, name


def test_pinning_an_untuned_name_is_refused_rather_than_ignored(sweep) -> None:
    """Silently pinning something nobody tunes would look like a controlled
    comparison and be none."""
    with pytest.raises(SystemExit) as caught:
        run([*BASE, "--steps", "300", "--trials", "2", "--seeds", "0",
             "--pin", "scalar_lr_multiplier=0.4"])
    assert "scalar_lr_multiplier" in str(caught.value)


def test_pinning_changes_which_configurations_are_drawn(sweep) -> None:
    """Removing a dimension must remove it from the random stream too, not
    draw it and overwrite it -- otherwise the pinned sweep explores the same
    projected points as the unpinned one and buys no search density."""
    free = astro_lab.draw_configs("muon", 4, {})
    pinned = astro_lab.draw_configs("muon", 4, {"scalar_lr_mult": 0.1})
    assert [c["lr"] for c in free] != [c["lr"] for c in pinned]


def test_every_optimizer_with_one_space_is_offered_the_same_configurations() -> None:
    """The paired-across-configurations table is only meaningful if this holds."""
    muon = astro_lab.draw_configs("muon", 5, {})
    normuon = astro_lab.draw_configs("normuon", 5, {})
    astro = astro_lab.draw_configs("astro", 5, {})
    assert muon == normuon == astro
    trust = astro_lab.draw_configs("astro_trust", 5, {})
    assert not any(astro_lab.same_config(a, b) for a, b in zip(muon, trust, strict=True))


def test_the_report_pairs_the_tuning_trials_across_configurations(sweep) -> None:
    run([*BASE, "--steps", "300", "--trials", "4", "--seeds", "0"])
    report = Path("astro_lab_report.md").read_text()
    assert "Across shared configurations" in report
    # The stub puts astro's optimum at lr=0.01 and muon's at 0.02, so astro
    # wins some configurations and loses others -- the row must exist either
    # way, and must count four shared configurations.
    row = [line for line in report.splitlines() if line.startswith("| `astro` |")]
    assert row, report
    assert row[0].split("|")[2].strip() == "4", row


def test_an_optimizer_in_its_own_space_contributes_no_false_pairs(sweep) -> None:
    """astro_trust's trial 3 is a different point in a different space than
    muon's trial 3; pairing on the index would compare unrelated runs."""
    run([*BASE, "--steps", "300", "--trials", "4", "--seeds", "0"])
    report = Path("astro_lab_report.md").read_text()
    section = report[report.index("Across shared configurations"):]
    assert "`astro_trust`" not in section, section


def test_a_longer_sweep_extends_a_shorter_one_rather_than_redrawing_it() -> None:
    """The two-session plan runs ``--trials 3`` and then ``--trials 6``. That
    only saves the first session's GPU time if trial k is the same point in
    both, so the second session reads t0..t2 back and runs only t3..t5."""
    short = astro_lab.draw_configs("muon", 3, {})
    long = astro_lab.draw_configs("muon", 6, {})
    assert long[:3] == short


def test_resuming_with_more_trials_keeps_the_earlier_ones(sweep) -> None:
    run([*BASE, "--steps", "300", "--trials", "2", "--seeds", "0"])
    sweep.clear()
    state = json.loads(Path("astro_lab_state.json").read_text())
    state["tuned"] = {}
    Path("astro_lab_state.json").write_text(json.dumps(state))

    run([*BASE, "--steps", "300", "--trials", "4", "--seeds", "0"])
    trials = json.loads(Path("astro_lab_state.json").read_text())["trials"]
    assert len(trials) == 12
    assert len(sweep) == 6, "extending the sweep re-ran trials it already had"


def test_the_run_prints_a_fingerprint_and_the_ranges_it_will_use(sweep, capsys) -> None:
    """A previous session's drawn values had to be reverse-engineered to work
    out whether the widened range was in the file that ran."""
    run([*BASE, "--steps", "300", "--trials", "2", "--seeds", "0",
         "--pin", "scalar_lr_mult=0.4369"])
    out = capsys.readouterr().out
    assert "astro_lab " in out.splitlines()[0]
    assert "scalar_lr_mult=*0.4369" in out
    low, high = astro_lab.MUON_LR
    assert f"lr=[{low:g},{high:g}]" in out


def test_the_run_counts_tuning_trials_not_just_evaluations(sweep, capsys) -> None:
    """Under ``--seeds 0`` the trials are the only runs, and the header used to
    announce "0 runs to go" at the start of a twelve-run sweep."""
    run([*BASE, "--steps", "900", "--trials", "4", "--seeds", "0"])
    out = capsys.readouterr().out
    assert "0 evaluations and 12 tuning trials to go" in out

    sweep.clear()
    state = json.loads(Path("astro_lab_state.json").read_text())
    state["tuned"] = {}
    Path("astro_lab_state.json").write_text(json.dumps(state))
    run([*BASE, "--steps", "900", "--trials", "6", "--seeds", "0"])
    assert "0 evaluations and 6 tuning trials to go" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Surviving a Colab runtime that is reclaimed mid-sweep
# ---------------------------------------------------------------------------


def test_a_colab_session_refuses_an_ephemeral_working_directory(sweep, monkeypatch) -> None:
    """A session ran eight 900-step trials from /content and was killed before
    it finished. All eight were lost with the runtime. A warning would have
    scrolled past in the first second of a two-hour cell."""
    monkeypatch.setattr(astro_lab, "in_colab", lambda: True)
    with pytest.raises(SystemExit) as caught:
        run([*BASE, "--steps", "300", "--trials", "2", "--seeds", "0"])
    message = str(caught.value)
    assert "does not survive" in message
    assert "--work-dir" in message
    assert sweep == [], "training started before the directory was checked"


def test_the_ephemeral_override_exists_for_a_throwaway_check(sweep, monkeypatch) -> None:
    monkeypatch.setattr(astro_lab, "in_colab", lambda: True)
    assert run([*BASE, "--steps", "300", "--trials", "1", "--seeds", "0",
                "--allow-ephemeral"]) == 0
    assert len(sweep) == 3


def test_work_dir_is_created_and_everything_lands_in_it(sweep, tmp_path) -> None:
    launched_from = Path.cwd()
    target = tmp_path / "drive" / "astro"
    assert not target.exists(), "the fixture already made it; the test proves nothing"

    assert run([*BASE, "--steps", "300", "--trials", "2", "--seeds", "0",
                "--work-dir", str(target)]) == 0

    assert (target / "astro_lab_state.json").exists()
    assert (target / "astro_lab_report.md").exists()
    assert not (launched_from / "astro_lab_state.json").exists(), \
        "state was written where the cell was launched, not in --work-dir"


def test_stop_after_bounds_a_cell_by_runs_not_by_the_clock(sweep) -> None:
    """--max-minutes does not help when what ends the cell is Colab reclaiming
    the runtime rather than the budget expiring."""
    assert run([*BASE, "--steps", "900", "--trials", "4", "--seeds", "0",
                "--stop-after", "5"]) == 0
    assert len(sweep) == 5
    trials = json.loads(Path("astro_lab_state.json").read_text())["trials"]
    assert len(trials) == 5


def test_repeating_a_stop_after_cell_finishes_the_sweep(sweep) -> None:
    for _ in range(3):
        state_file = Path("astro_lab_state.json")
        if state_file.exists():
            state = json.loads(state_file.read_text())
            state["tuned"] = {}
            state_file.write_text(json.dumps(state))
        run([*BASE, "--steps", "900", "--trials", "4", "--seeds", "0",
             "--stop-after", "5"])
    trials = json.loads(Path("astro_lab_state.json").read_text())["trials"]
    assert len(trials) == 12, sorted(trials)
    assert len(sweep) == 12, "a repeated cell re-ran work it had already done"


def test_a_stopped_cell_still_writes_the_report(sweep) -> None:
    run([*BASE, "--steps", "900", "--trials", "4", "--seeds", "0",
         "--stop-after", "2"])
    assert Path("astro_lab_report.md").exists()


def test_the_cost_estimator_reads_tuning_trials_too(sweep, monkeypatch) -> None:
    """Under --seeds 0 there are no evaluation runs at all, so an estimator
    that reads only ``runs`` stays on its generic T4 guess for the whole
    session -- the session where the budget guard matters most."""
    run([*BASE, "--steps", "900", "--trials", "1", "--seeds", "0"])
    state = json.loads(Path("astro_lab_state.json").read_text())
    assert state["runs"] == {}
    assert state["trials"], "nothing recorded to estimate from"
    # The stub reports one second per step; the generic fallback is 1.15.
    for entry in state["trials"].values():
        assert entry["seconds"] == 900

    sweep.clear()
    state["tuned"] = {}
    Path("astro_lab_state.json").write_text(json.dumps(state))

    clock = {"now": 0.0}
    monkeypatch.setattr(astro_lab.time, "perf_counter", lambda: clock["now"])
    real_train = astro_lab.train_once

    def timed(name, config, seed, **kwargs):
        clock["now"] += kwargs["steps"] * 1.0     # 15 minutes for 900 steps
        return real_train(name, config, seed, **kwargs)

    monkeypatch.setattr(astro_lab, "train_once", timed)

    # The recorded trials say 1.0 s/step, so the guard predicts 16.5 minutes
    # with its 10% headroom and one run fits a 17-minute budget. The generic
    # T4 fallback of 1.15 s/step predicts 19.0 and would refuse to start at
    # all -- so this budget separates the two.
    run([*BASE, "--steps", "900", "--trials", "2", "--seeds", "0",
         "--max-minutes", "17"])
    assert len(sweep) == 1, [c["name"] for c in sweep]


# ---------------------------------------------------------------------------
# The shared grid runs configuration-major, so any stopping point is balanced
# ---------------------------------------------------------------------------


def _per_optimizer(names: tuple[str, ...]) -> dict[str, int]:
    trials = json.loads(Path("astro_lab_state.json").read_text())["trials"]
    return {n: sum(1 for k in trials if k.split("|")[2] == n) for n in names}


NAMES = ("muon", "astro", "astro_trust")


def test_a_truncated_sweep_is_balanced_across_optimizers(sweep) -> None:
    """A real session stopped after six runs and produced a report comparing
    three baselines with both ASTRO columns simply absent -- because the sweep
    walked optimizers in the outer loop and ASTRO was listed last."""
    run([*BASE, "--steps", "900", "--trials", "4", "--seeds", "0",
         "--stop-after", "5"])
    per = _per_optimizer(NAMES)
    assert sum(per.values()) == 5
    assert max(per.values()) - min(per.values()) <= 1, per


def test_every_optimizer_appears_in_a_truncated_report(sweep) -> None:
    run([*BASE, "--steps", "900", "--trials", "4", "--seeds", "0",
         "--stop-after", "5"])
    report = Path("astro_lab_report.md").read_text()
    assert "Across shared configurations" in report
    assert "`astro`" in report, report


def test_the_report_is_written_from_the_first_configuration_onward(sweep) -> None:
    """Not only when the sweep finishes -- a killed session must still leave a
    readable table beside its numbers."""
    run([*BASE, "--steps", "900", "--trials", "6", "--seeds", "0",
         "--stop-after", "3"])
    report = Path("astro_lab_report.md").read_text()
    assert "| `astro` |" in report, report
    assert _per_optimizer(NAMES) == {"muon": 1, "astro": 1, "astro_trust": 1}


def test_stopping_mid_configuration_leaves_at_most_one_incomplete(sweep) -> None:
    run([*BASE, "--steps", "900", "--trials", "4", "--seeds", "0",
         "--stop-after", "7"])
    per = _per_optimizer(NAMES)
    assert sorted(per.values()) == [2, 2, 3], per


def test_selection_reads_the_state_file_not_a_running_variable(sweep) -> None:
    """One session or five must select the same configuration."""
    run([*BASE, "--steps", "900", "--trials", "4", "--seeds", "0"])
    whole = json.loads(Path("astro_lab_state.json").read_text())["tuned"]

    Path("astro_lab_state.json").unlink()
    sweep.clear()
    for cap in (2, 4, 6, 8, 10, 12):
        state_file = Path("astro_lab_state.json")
        if state_file.exists():
            state = json.loads(state_file.read_text())
            state["tuned"] = {}
            state_file.write_text(json.dumps(state))
        run([*BASE, "--steps", "900", "--trials", "4", "--seeds", "0",
             "--stop-after", str(cap)])
    piecewise = json.loads(Path("astro_lab_state.json").read_text())["tuned"]
    assert piecewise == whole
    assert len(sweep) == 12, "a piecewise sweep repeated work"


def test_raising_trials_draws_the_new_configurations(sweep) -> None:
    """The documented ladder is --trials 2, then 3, then 4, in separate Colab
    cells. Gating the sweep on "is this optimizer already selected" made every
    cell after the first a silent no-op: each optimizer had a selection, so the
    extra configurations were never drawn and the session did nothing."""
    run([*BASE, "--steps", "900", "--trials", "2", "--seeds", "0"])
    assert _per_optimizer(NAMES) == {"muon": 2, "astro": 2, "astro_trust": 2}

    sweep.clear()
    run([*BASE, "--steps", "900", "--trials", "3", "--seeds", "0"])
    assert len(sweep) == 3, "raising --trials ran nothing"
    assert _per_optimizer(NAMES) == {"muon": 3, "astro": 3, "astro_trust": 3}

    sweep.clear()
    run([*BASE, "--steps", "900", "--trials", "5", "--seeds", "0"])
    assert len(sweep) == 6
    assert _per_optimizer(NAMES) == {"muon": 5, "astro": 5, "astro_trust": 5}


def test_the_ladder_needs_no_manual_state_editing(sweep) -> None:
    for trials in range(1, 5):
        run([*BASE, "--steps", "900", "--trials", str(trials), "--seeds", "0"])
    assert len(sweep) == 12
    assert _per_optimizer(NAMES) == {"muon": 4, "astro": 4, "astro_trust": 4}
    report = Path("astro_lab_report.md").read_text()
    assert "| `astro` | 4 |" in report, report


def test_evaluation_does_not_re_run_a_completed_grid(sweep) -> None:
    run([*BASE, "--steps", "900", "--trials", "3", "--seeds", "0"])
    sweep.clear()
    run([*BASE, "--steps", "900", "--trials", "3", "--seeds", "2"])
    assert not [c for c in sweep if c["seed"] == 0], "re-ran the grid"
    assert len(sweep) == 6


# ---------------------------------------------------------------------------
# Running the file you meant to run
# ---------------------------------------------------------------------------


def test_a_duplicate_download_is_refused(sweep, monkeypatch, tmp_path) -> None:
    """A browser that downloads the same name twice writes "astro_lab (1).py".
    A session was lost running one: it predated --pin entirely, so the sweep it
    would have run was not the sweep that was asked for."""
    stale = tmp_path / "astro_lab (1).py"
    stale.write_bytes(b"# whatever\n")
    monkeypatch.setattr(astro_lab, "SOURCE", stale)
    with pytest.raises(SystemExit) as caught:
        run([*BASE, "--steps", "300", "--trials", "1", "--seeds", "0"])
    assert "duplicate download" in str(caught.value)
    assert sweep == []


def test_expect_refuses_a_version_nobody_asked_for(sweep) -> None:
    with pytest.raises(SystemExit) as caught:
        run([*BASE, "--steps", "300", "--trials", "1", "--seeds", "0",
             "--expect", "deadbeef"])
    assert "deadbeef" in str(caught.value)
    assert sweep == []


def test_expect_passes_on_the_real_digest(sweep, capsys) -> None:
    import hashlib

    digest = hashlib.sha256(astro_lab.SOURCE.read_bytes()).hexdigest()
    assert run([*BASE, "--steps", "300", "--trials", "1", "--seeds", "0",
                "--expect", digest[:8]]) == 0
    assert len(sweep) == 3
    assert digest[:12] in capsys.readouterr().out


def test_the_first_line_names_the_file_that_ran(sweep, capsys) -> None:
    run([*BASE, "--steps", "300", "--trials", "1", "--seeds", "0"])
    first = capsys.readouterr().out.splitlines()[0]
    assert astro_lab.SOURCE.name in first, first


def test_an_evaluation_run_from_another_protocol_is_not_reused(sweep) -> None:
    """A state file carried across protocols held muon and normuon evaluation
    runs selected by an abandoned tuning design. The resume check keyed on
    (size, steps, optimizer, seed) alone, so those would have been reused and
    compared against optimizers evaluated under the current protocol -- one
    seed table built from two different experiments."""
    run([*BASE, "--steps", "900", "--trials", "2", "--seeds", "1"])
    state = json.loads(Path("astro_lab_state.json").read_text())
    assert set(state["runs"]) == {f"124M|900|{n}|100" for n in NAMES}

    # Rewrite muon's stored run as if an older protocol had selected a
    # different learning rate, and clear nothing else.
    stale = dict(state["runs"]["124M|900|muon|100"])
    stale["config"] = dict(stale["config"], lr=stale["config"]["lr"] * 3)
    stale["value"] = 9.99
    state["runs"]["124M|900|muon|100"] = stale
    Path("astro_lab_state.json").write_text(json.dumps(state))

    sweep.clear()
    run([*BASE, "--steps", "900", "--trials", "2", "--seeds", "1"])
    assert [c["name"] for c in sweep] == ["muon"], [c["name"] for c in sweep]
    after = json.loads(Path("astro_lab_state.json").read_text())
    assert after["runs"]["124M|900|muon|100"]["value"] != 9.99


def test_a_matching_evaluation_run_is_still_reused(sweep) -> None:
    run([*BASE, "--steps", "900", "--trials", "2", "--seeds", "1"])
    sweep.clear()
    run([*BASE, "--steps", "900", "--trials", "2", "--seeds", "1"])
    assert sweep == [], "re-ran an evaluation at the configuration it recorded"
