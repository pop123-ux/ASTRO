"""Regression tests for the ablation campaign definition."""

from scripts.ablation_suite import ASTRO, BASELINES, apply_shard, build_suite


def test_optimizer_surface():
    assert len(BASELINES) == 4
    assert len(ASTRO) == 17
    assert len(set(BASELINES + ASTRO)) == 21


def test_suite_sizes_are_deterministic():
    # 18 dense optimizers/variants * 6 LR * 5 WD * 5 scalar + 8 tuner controls.
    assert len(build_suite("primary")) == 2708
    # 21 methods * 3 horizons * 3 seeds + 5 methods * 7 LR factors * 3 seeds.
    assert len(build_suite("mechanisms")) == 252
    assert len(build_suite("robustness")) == 252
    assert len(build_suite("stability")) == 192


def test_job_ids_are_unique():
    for suite in ("primary", "mechanisms", "robustness", "stability"):
        jobs = build_suite(suite)
        ids = [job.job_id for job in jobs]
        assert len(ids) == len(set(ids))


def test_sharding_is_complete_and_disjoint():
    jobs = build_suite("mechanisms")
    shards = [apply_shard(jobs, f"{part}/3") for part in range(3)]
    ids = [job.job_id for part in shards for job in part]
    assert len(ids) == len(jobs)
    assert len(ids) == len(set(ids))


def test_seed_is_explicit():
    for job in build_suite("mechanisms")[:30]:
        assert job.seed in (100, 101, 102)
