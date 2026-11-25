"""Tests for deterministic high-speed camera auto exposure."""

import numpy as np
import pytest

from openflight.camera.auto_exposure import (
    EXPOSURE_STEPS,
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


def start_at(index):
    """(exposure_us, gain) of the ladder step at ``index``."""
    step = EXPOSURE_STEPS[index]
    return step.exposure_us, step.gain


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


def test_every_ladder_step_is_brighter_than_the_one_before():
    # A step with the same brightness as its predecessor adds blur or noise but no light.
    signals = [step.signal for step in EXPOSURE_STEPS]
    assert all(later > earlier for earlier, later in zip(signals, signals[1:]))


def test_no_ladder_step_is_more_than_two_times_brighter_than_the_last():
    signals = [step.signal for step in EXPOSURE_STEPS]
    assert max(later / earlier for earlier, later in zip(signals, signals[1:])) <= 2.0 + 1e-9


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
    # Never "too dark" when the ball is visible; a dim scene is only "marginal" (accepted).
    image = np.full((200, 320), 58, dtype=np.uint8)
    yy, xx = np.mgrid[:200, :320]
    image[(xx - 160) ** 2 + (yy - 130) ** 2 <= 6 * 6] = 220

    result = measure_exposure(image)

    assert result.status == "marginal"
    assert result.acceptable is True
    assert result.peak >= 200.0


def ball_on_background(background, ball):
    """Impact-zone frame: a flat background with one small ball (r=8 px)."""
    image = np.full((200, 320), background, dtype=np.uint8)
    yy, xx = np.mgrid[:200, :320]
    image[(xx - 160) ** 2 + (yy - 130) ** 2 <= 8 * 8] = ball
    return image


def test_measure_exposure_flags_a_ball_washed_out_against_a_bright_background():
    # White on white: the background is not blown out (median 200, nothing clipped) but the
    # ball is only 35 gray levels above it, so it cannot be told apart.
    result = measure_exposure(ball_on_background(200, 235))

    assert result.status == "too_bright"
    assert result.recommendation == "darker"
    assert result.ball_contrast == pytest.approx(35.0)
    assert "washed out" in result.message


def test_measure_exposure_keeps_a_distinct_ball_on_a_bright_background():
    result = measure_exposure(ball_on_background(170, 250))

    assert result.status == "good"
    assert result.ball_contrast == pytest.approx(80.0)


def test_measure_exposure_is_not_good_when_the_ball_is_near_the_detection_floor():
    # Ball at 135 on a 60 background: plenty of contrast, but only just above the level the
    # ball and club masks need downstream (captures at this level were mostly rejected as
    # low light), so keep brightening.
    result = measure_exposure(ball_on_background(60, 135))

    assert result.status == "marginal"
    assert result.recommendation == "brighter"


def test_measure_exposure_is_good_when_the_ball_clears_the_detection_floor_with_margin():
    result = measure_exposure(ball_on_background(100, 200))

    assert result.status == "good"


def test_measure_exposure_is_not_good_when_the_ball_barely_stands_out():
    # Mid-gray scene: nothing clipped or washed out, but the ball is only 30 levels brighter.
    result = measure_exposure(ball_on_background(100, 130))

    assert result.status == "marginal"
    assert result.recommendation == "brighter"


def test_measure_exposure_recommends_darker_for_low_contrast_on_a_bright_scene():
    # Contrast 45: above the washed-out limit, below the good limit, with a bright median.
    result = measure_exposure(ball_on_background(160, 205))

    assert result.status == "marginal"
    assert result.recommendation == "darker"


def test_measure_exposure_is_not_good_when_the_scene_is_dim_even_if_the_ball_is_bright():
    # Real range frame at 500us/12: ball 237 but the surrounding scene (and the club) sits near
    # gray 60-90, below what the club mask needs, so every shot was rejected as low light.
    result = measure_exposure(ball_on_background(66, 237))

    assert result.status == "marginal"
    assert result.recommendation == "brighter"
    assert result.sample_available is True
    assert result.acceptable is True


def test_measure_exposure_is_good_when_the_scene_is_lit_well_enough_for_the_club():
    # Real range frame at 800us/18: scene p90 ~100-105, club visible.
    result = measure_exposure(ball_on_background(104, 250))

    assert result.status == "good"


def test_measure_exposure_is_not_good_when_the_scene_is_just_below_the_club_threshold():
    # Real range frame at 650us/15: scene p90 96 locked here and only 2 of 4 club paths and 0 of 4
    # horizontal launch angles registered, so it must keep brightening.
    result = measure_exposure(ball_on_background(96, 250))

    assert result.status == "marginal"
    assert result.recommendation == "brighter"


def test_quality_score_rewards_a_brighter_scene_when_the_ball_is_already_bright():
    dim_scene = observation("marginal", "brighter", median=58.0, p90=66.0, peak=161.0)
    lit_scene = observation("good", "hold", median=91.0, p90=105.0, peak=161.0)

    assert exposure_quality_score(lit_scene) > exposure_quality_score(dim_scene)


def test_refinement_brightens_when_only_the_scene_is_dim():
    policy = AutoExposurePolicy(fps=488.0)
    dim = observation("marginal", "brighter", median=58.0, p90=66.0, peak=162.0)
    lit = observation("good", "hold", median=91.0, p90=100.0, peak=165.0)

    decisions, final = run_policy(policy, [dim, lit], start=start_at(11))

    assert decisions[0].target == EXPOSURE_STEPS[12]
    assert decisions[-1].status == "ready"
    assert final == start_at(12)


def test_observation_reports_ball_contrast_in_its_payload():
    result = measure_exposure(ball_on_background(60, 200))

    assert result.to_dict()["ball_contrast"] == pytest.approx(140.0)


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
    start_index = next(
        index
        for index, step in enumerate(EXPOSURE_STEPS)
        if (step.exposure_us, step.gain) == (250, 4.0)
    )
    assert decision.target.signal > EXPOSURE_STEPS[start_index + 1].signal
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
        "too_bright", "darker", median=150.0, peak=255.0, clipped_pct=22.0
    )
    darkest = EXPOSURE_STEPS[0]

    decision = policy.evaluate(brightest_scene, exposure_us=darkest.exposure_us, gain=darkest.gain)

    assert decision.status == "ready"


def test_darkest_step_rejects_a_washed_out_frame_where_the_ball_is_lost():
    policy = AutoExposurePolicy(fps=488.0)
    washed_out = observation("too_bright", "darker", median=225.0, peak=245.0, clipped_pct=2.0)
    darkest = EXPOSURE_STEPS[0]

    decision = policy.evaluate(washed_out, exposure_us=darkest.exposure_us, gain=darkest.gain)

    assert decision.status == "lighting_required"
    assert decision.analysis_eligible is False
    assert "overexposed" in decision.message


def test_marginal_start_refines_and_stops_at_the_first_good_frame():
    policy = AutoExposurePolicy(fps=488.0)
    marginal = observation("marginal", "brighter", median=60.0, peak=100.0)
    good = observation("good", "hold", median=62.0, peak=185.0)

    decisions, final = run_policy(policy, [marginal, good], start=start_at(5))

    assert decisions[0].status == "adjusting"
    assert decisions[0].target == EXPOSURE_STEPS[6]
    assert decisions[-1].status == "ready"
    assert final == start_at(6)
    assert policy.startup is False


def test_good_start_probes_darker_then_brighter_and_restores_the_best_step():
    policy = AutoExposurePolicy(fps=488.0)
    start = start_at(8)
    good = observation("good", "hold", median=100.0, peak=210.0)
    worse = observation("good", "hold", median=100.0, peak=150.0)

    decisions, final = run_policy(policy, [good, worse, worse, good], start=start)

    probes = [decision.target for decision in decisions if decision.target is not None]
    assert probes[0] == EXPOSURE_STEPS[7]  # darker first
    assert probes[1] == EXPOSURE_STEPS[9]  # then brighter
    assert decisions[-1].status == "ready"
    assert final == start
    assert policy.startup is False


def test_refinement_follows_a_shorter_exposure_that_scores_better():
    policy = AutoExposurePolicy(fps=488.0)
    start = start_at(9)
    better = observation("good", "hold", median=80.0, peak=215.0)
    good = observation("good", "hold", median=120.0, peak=215.0)

    # start, darker probe (better), darker again (better), darker again (worse -> restore
    # the best step), then the restored step confirms and locks.
    decisions, final = run_policy(policy, [good, better, better, good, better], start=start)

    assert decisions[-1].status == "ready"
    assert final[0] < start[0]


def test_refinement_keeps_brightening_through_marginal_frames_until_the_ball_is_good():
    # Each step up costs more gain/exposure than its small peak gain is worth on its own, but
    # the ball is still below the "good" peak, so the climb must continue until it clears it.
    policy = AutoExposurePolicy(fps=488.0)
    first = observation("marginal", "brighter", median=50.0, peak=127.0)
    second = observation("marginal", "brighter", median=55.0, peak=140.0)
    third = observation("good", "hold", median=60.0, peak=172.0)

    decisions, final = run_policy(policy, [first, second, third], start=start_at(10))

    assert decisions[-1].status == "ready"
    assert final == start_at(12)


def test_refinement_climbs_to_the_end_of_the_ladder_when_the_ball_is_far_below_good():
    # A very dim scene: each step up adds blur and gain, but the ball is nowhere near bright
    # enough for the club and ball masks, so the climb must not stop early.
    policy = AutoExposurePolicy(fps=488.0)
    dim = [
        observation("marginal", "brighter", median=17.0, peak=61.0),
        observation("marginal", "brighter", median=23.0, peak=81.0),
        observation("marginal", "brighter", median=28.0, peak=97.0),
    ]

    decisions, final = run_policy(policy, dim, start=start_at(len(EXPOSURE_STEPS) - 3))

    assert decisions[-1].status == "ready"
    assert final == start_at(-1)


def test_refinement_is_bounded_by_the_step_budget():
    policy = AutoExposurePolicy(fps=488.0, refine_max_steps=2)
    start = start_at(6)
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
