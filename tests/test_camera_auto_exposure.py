"""Tests for deterministic high-speed camera auto exposure."""

import numpy as np
import pytest

from openflight.camera.auto_exposure import (
    EXPOSURE_STEPS,
    GAIN_COMFORT_LIMIT,
    AutoExposurePolicy,
    ExposureObservation,
    exposure_quality_score,
    exposure_steps_for_fps,
    measure_exposure,
    motion_blur_risk,
)


def observation(  # pylint: disable=too-many-arguments
    status, recommendation, *, median=50.0, p90=100.0, peak=None, clipped_pct=0.0
):
    """Build one deterministic policy observation."""
    return ExposureObservation(
        sample_available=True,
        status=status,
        recommendation=recommendation,
        message=status,
        p10=10.0,
        median=median,
        p90=p90,
        contrast=p90 - 10.0,
        clipped_pct=clipped_pct,
        dark_pct=0.0,
        peak=peak,
    )


def step_index(exposure_us, gain):
    """Index of one exact ladder step."""
    return next(
        index
        for index, step in enumerate(EXPOSURE_STEPS)
        if (step.exposure_us, step.gain) == (exposure_us, gain)
    )


def run_policy(policy, observations, *, start):
    """Feed observations to a policy, following its targets; return the decisions."""
    exposure_us, gain = start
    decisions = []
    for obs in observations:
        decision = policy.evaluate(obs, exposure_us=exposure_us, gain=gain)
        decisions.append(decision)
        if decision.target is not None:
            exposure_us, gain = decision.target.exposure_us, decision.target.gain
        if decision.status in {"ready", "lighting_required"}:
            break
    return decisions, (exposure_us, gain)


def test_ladder_keeps_gain_at_comfort_limit_until_750us():
    first_high_gain = next(
        index for index, step in enumerate(EXPOSURE_STEPS) if step.gain > GAIN_COMFORT_LIMIT
    )
    assert EXPOSURE_STEPS[first_high_gain - 1] == EXPOSURE_STEPS[step_index(750, 10.0)]
    assert all(step.gain <= GAIN_COMFORT_LIMIT for step in EXPOSURE_STEPS[:first_high_gain])


def test_exposure_ladder_is_monotonic_and_respects_frame_period():
    assert [step.signal for step in EXPOSURE_STEPS] == sorted(
        step.signal for step in EXPOSURE_STEPS
    )
    assert exposure_steps_for_fps(488.0) == EXPOSURE_STEPS
    assert max(step.exposure_us for step in exposure_steps_for_fps(1000.0)) < 1000
    with pytest.raises(ValueError, match="positive"):
        exposure_steps_for_fps(0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(18, "too_dark"), (110, "good"), (252, "too_bright")],
)
def test_measure_exposure_classifies_impact_zone(value, expected):
    image = np.full((200, 320), value, dtype=np.uint8)
    if expected == "good":
        image[100:175, 90:230] = np.tile(
            np.linspace(45, 185, 140, dtype=np.uint8),
            (75, 1),
        )

    result = measure_exposure(image)

    assert result.status == expected
    assert result.sample_available is True


def test_measure_exposure_accepts_dark_background_with_visible_ball():
    image = np.full((200, 320), 58, dtype=np.uint8)
    yy, xx = np.mgrid[:200, :320]
    image[(xx - 160) ** 2 + (yy - 130) ** 2 <= 6 * 6] = 220

    result = measure_exposure(image)

    assert result.status == "good"
    assert result.peak >= 200.0


def test_measure_exposure_flags_nearly_black_view():
    result = measure_exposure(np.zeros((200, 320), dtype=np.uint8))

    assert result.status == "too_dark"
    assert "lens cover" in result.message


def test_startup_dark_scene_jumps_more_than_one_step():
    policy = AutoExposurePolicy(fps=488.0)

    decision = policy.evaluate(
        observation("too_dark", "brighter", median=18.0, p90=40.0),
        exposure_us=250,
        gain=4.0,
    )

    assert decision.status == "adjusting"
    assert decision.target is not None
    assert decision.target.signal > EXPOSURE_STEPS[4].signal
    assert decision.analysis_eligible is False


def test_startup_bright_scene_jumps_darker():
    policy = AutoExposurePolicy(fps=488.0)

    decision = policy.evaluate(
        observation("too_bright", "darker", median=240.0, p90=255.0),
        exposure_us=800,
        gain=18.0,
    )

    assert decision.target is not None
    assert decision.target.signal < 800 * 18


def test_acceptable_startup_observation_marks_analysis_ready():
    policy = AutoExposurePolicy(fps=488.0, probe_good_neighbors=False)

    decision = policy.evaluate(
        observation("good", "hold", median=100.0),
        exposure_us=500,
        gain=12.0,
    )

    assert decision.status == "ready"
    assert decision.analysis_eligible is True
    assert not decision.should_apply


def test_ready_setting_stays_locked_until_the_next_startup():
    policy = AutoExposurePolicy(fps=488.0, refine_max_steps=0)
    ready = policy.evaluate(
        observation("good", "hold"),
        exposure_us=500,
        gain=12.0,
    )

    changed_scene = policy.evaluate(
        observation("too_dark", "brighter", median=18.0, p90=40.0),
        exposure_us=500,
        gain=12.0,
    )
    policy.reset()
    next_startup = policy.evaluate(
        observation("too_dark", "brighter", median=18.0, p90=40.0),
        exposure_us=500,
        gain=12.0,
    )

    assert changed_scene is ready
    assert changed_scene.status == "ready"
    assert not changed_scene.should_apply
    assert next_startup.status == "adjusting"
    assert next_startup.should_apply
    assert policy.startup is True


def test_ladder_limit_stays_locked_until_the_next_startup():
    policy = AutoExposurePolicy(fps=488.0)
    darkest = EXPOSURE_STEPS[-1]

    failed = policy.evaluate(
        observation("too_dark", "brighter", median=10.0, p90=20.0),
        exposure_us=darkest.exposure_us,
        gain=darkest.gain,
    )
    locked = policy.evaluate(
        observation("marginal", "brighter", median=40.0, p90=95.0),
        exposure_us=darkest.exposure_us,
        gain=darkest.gain,
    )
    policy.reset()
    recovered = policy.evaluate(
        observation("marginal", "brighter", median=40.0, p90=95.0),
        exposure_us=darkest.exposure_us,
        gain=darkest.gain,
    )

    assert failed.status == "lighting_required"
    assert failed.analysis_eligible is False
    assert "add or redirect light" in failed.message
    assert locked is failed
    assert recovered.status == "ready"
    assert recovered.analysis_eligible is True


def test_startup_attempt_limit_requires_lighting_change():
    policy = AutoExposurePolicy(fps=488.0, startup_max_adjustments=1)
    dark = observation("too_dark", "brighter", median=8.0, p90=20.0, peak=22.0)
    first = policy.evaluate(dark, exposure_us=250, gain=4.0)

    failed = policy.evaluate(
        dark,
        exposure_us=first.target.exposure_us,
        gain=first.target.gain,
    )

    assert failed.status == "lighting_required"
    assert not failed.should_apply


@pytest.mark.parametrize(
    ("exposure_us", "expected"),
    [(500, "low"), (650, "elevated"), (1000, "high")],
)
def test_motion_blur_risk(exposure_us, expected):
    assert motion_blur_risk(exposure_us) == expected


def test_startup_attempt_limit_accepts_a_tolerable_dim_frame():
    policy = AutoExposurePolicy(fps=488.0, startup_max_adjustments=1)
    dim = observation("too_dark", "brighter", median=18.0, p90=40.0, peak=45.0)
    first = policy.evaluate(dim, exposure_us=250, gain=4.0)

    settled = policy.evaluate(dim, exposure_us=first.target.exposure_us, gain=first.target.gain)

    assert settled.status == "ready"
    assert settled.analysis_eligible is True
    assert "Best available" in settled.message


def test_darkest_step_accepts_a_washed_out_but_usable_frame():
    policy = AutoExposurePolicy(fps=488.0)
    brightest_scene = observation(
        "too_bright", "darker", median=235.0, peak=255.0, clipped_pct=22.0
    )
    darkest = EXPOSURE_STEPS[0]

    decision = policy.evaluate(brightest_scene, exposure_us=darkest.exposure_us, gain=darkest.gain)

    assert decision.status == "ready"


def test_marginal_start_refines_and_stops_at_the_first_good_frame():
    policy = AutoExposurePolicy(fps=488.0)
    marginal = observation("marginal", "brighter", median=60.0, peak=100.0)
    good = observation("good", "hold", median=62.0, peak=185.0)

    decisions, final = run_policy(policy, [marginal, good], start=(200, 9.0))

    assert decisions[0].status == "adjusting"
    assert decisions[0].target == EXPOSURE_STEPS[step_index(200, 9.0) + 1]
    assert decisions[-1].status == "ready"
    assert final == (EXPOSURE_STEPS[step_index(200, 9.0) + 1].exposure_us, 10.0)
    assert policy.startup is False


def test_good_start_probes_darker_then_brighter_and_restores_the_best_step():
    policy = AutoExposurePolicy(fps=488.0)
    start = (325, 10.0)
    good = observation("good", "hold", median=100.0, peak=210.0)
    worse = observation("good", "hold", median=100.0, peak=150.0)

    decisions, final = run_policy(policy, [good, worse, worse, good], start=start)

    probes = [decision.target for decision in decisions if decision.target is not None]
    assert probes[0] == EXPOSURE_STEPS[step_index(*start) - 1]  # darker first
    assert probes[1] == EXPOSURE_STEPS[step_index(*start) + 1]  # then brighter
    assert decisions[-1].status == "ready"
    assert final == start
    assert policy.startup is False


def test_refinement_follows_a_shorter_exposure_that_scores_better():
    policy = AutoExposurePolicy(fps=488.0)
    start = (450, 10.0)
    better = observation("good", "hold", median=80.0, peak=215.0)
    good = observation("good", "hold", median=120.0, peak=215.0)

    # start, darker probe (better), darker again (better), darker again (worse -> restore
    # the best step), then the restored step confirms and locks.
    decisions, final = run_policy(policy, [good, better, better, good, better], start=start)

    assert decisions[-1].status == "ready"
    assert final[0] < start[0]


def test_refinement_is_bounded_by_the_step_budget():
    policy = AutoExposurePolicy(fps=488.0, refine_max_steps=2)
    start = (150, 6.5)
    improving = [
        observation("marginal", "brighter", median=60.0, peak=70.0 + 15.0 * index)
        for index in range(10)
    ]

    decisions, _ = run_policy(policy, improving, start=start)

    assert decisions[-1].status == "ready"
    assert sum(1 for decision in decisions if decision.target is not None) <= 3


def test_quality_score_prefers_short_exposure_and_low_gain():
    good = observation("good", "hold", median=100.0, peak=210.0)

    assert exposure_quality_score(good, 250, 10.0) > exposure_quality_score(good, 500, 10.0)
    assert exposure_quality_score(good, 350, 10.0) > exposure_quality_score(good, 350, 16.0)
    assert exposure_quality_score(good) > exposure_quality_score(
        observation("good", "hold", median=100.0, peak=210.0, clipped_pct=5.0)
    )
