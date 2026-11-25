"""Deterministic exposure selection for the high-speed OV9281 camera."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

ExposureQualityStatus = Literal[
    "good",
    "marginal",
    "too_dark",
    "too_bright",
    "unavailable",
]
ExposureRecommendation = Literal["brighter", "darker", "hold"]
AutoExposureStatus = Literal[
    "ready",
    "adjusting",
    "lighting_required",
    "unavailable",
]

# ---------------------------------------------------------------------------
# Tunable thresholds (8-bit grayscale, measured inside the impact-zone ROI)
#
# The ROI is mostly background (mat/turf/wall) with a small ball and clubhead.
# Whole-ROI statistics such as the median and p90 therefore describe the
# background, not the ball. A dark, uncluttered background is actually ideal
# for IR capture, so darkness is judged by the brightest meaningful feature
# (peak) rather than by median/contrast, and brightness is judged by how much
# of the ROI is blown out.
# ---------------------------------------------------------------------------

# Percentile used as the "bright feature" (ball / club highlight) signal. The ball covers only
# about 0.75-0.95% of the ROI in real frames, so the 99th percentile sits on its dim edge
# (reading 129 when the ball peaks at 213). 99.7 lands inside the ball while ignoring a few
# hot pixels. Raise it if the ball ever looks smaller than ~0.3% of the ROI.
BRIGHT_FEATURE_PERCENTILE = 99.7

# too_dark: nothing in the ROI is bright enough to image a ball or clubhead.
NEARLY_BLACK_PEAK = 25.0
TOO_DARK_PEAK = 60.0

# too_bright: a large part of the ROI is saturated, or the whole scene is washed out.
TOO_BRIGHT_CLIPPED_PCT = 20.0
TOO_BRIGHT_MEDIAN = 225.0

# Ball contrast = peak (the ball) minus the ROI median (the background). The ball has to stand
# out from the background to be tracked. On a bright scene a ball that does not stand out is
# "white on white" even when little is clipped, so it counts as too bright:
#   - WASHED_OUT_MEDIAN: backgrounds at or above this level are bright enough for that to matter;
#   - MIN_BALL_CONTRAST: below this the ball cannot be told apart from the background.
# Real frames (OV9281, 320x200, ball ~1% of the ROI) give 70-105 for good exposures and fall to
# ~58 as gain pushes the background toward 200, so 40 only fires on clearly lost balls.
WASHED_OUT_MEDIAN = 150.0
MIN_BALL_CONTRAST = 40.0

# good: comfortably exposed. Anything between good and the failure limits is "marginal"
# and is still accepted for analysis.
# The ball must clear the downstream detection floors with margin: club_delivery needs ball pixels
# above BALL_THRESHOLD_FLOOR (120) and a moving-bright club mask above max(110, 0.85 x ball peak).
# Range captures at ball peak 174 passed club path and AOA, while captures at 116 were mostly
# rejected as low light even though calibration rated them good, so require clear headroom.
GOOD_PEAK_MIN = 150.0
GOOD_MEDIAN_MAX = 190.0
# The scene (not just the ball) must be lit well enough to see the club. The ball is ~1% of the
# ROI, so the ROI p90 describes the background, ground and clubhead. Range captures with a ball
# peak of 237 but scene p90 ~66 (club region peaking at 89) were rejected on every shot because the
# club mask needs pixels above 110, while scene p90 ~100 (club region 150) was detected on every
# shot. A brighter ball alone does not make a frame good; the scene has to reach this level too.
GOOD_SCENE_P90_MIN = 90.0
GOOD_CLIPPED_MAX_PCT = 4.0
# A "good" frame also needs the ball to stand out; between MIN_BALL_CONTRAST and this it is
# marginal. Real good exposures measured 70-105.
GOOD_MIN_CONTRAST = 50.0

# Last-resort acceptance. When the controller cannot move any further (end of the
# ladder) or runs out of startup adjustments, an image that is imperfect but not
# catastrophic is accepted instead of reporting "lighting required".
TOLERABLE_MIN_PEAK = 40.0
TOLERABLE_MAX_CLIPPED_PCT = 35.0
TOLERABLE_MAX_MEDIAN = 240.0

# Startup targeting: aim the bright feature at this level when brightening, and
# the scene median at this level when darkening.
BRIGHTER_TARGET_PEAK = 170.0
DARKER_TARGET_MEDIAN = 110.0

# Refinement ("hill climb") after the first acceptable frame. The controller probes
# neighboring ladder steps and keeps moving while the quality score improves by at least
# REFINE_MIN_IMPROVEMENT:
#   - first frame marginal: probe one step in the recommended direction, and stop climbing as
#     soon as a frame rates "good" (the good-enough threshold);
#   - first frame good: probe one step darker and one step brighter to confirm no neighbor is
#     clearly better, and keep climbing in whichever direction helps.
# It always ends on the best step seen. REFINE_MAX_STEPS caps the total number of probe moves.
REFINE_MAX_STEPS = 4
REFINE_MIN_IMPROVEMENT = 0.05

# Quality score, higher is better. Two terms each reach 1.0: ball brightness (SCORE_PEAK_TARGET)
# and ball-to-background contrast, peak minus median (SCORE_CONTRAST_TARGET). Contrast is what
# falls as gain rises (the background washes out while the ball stays near white). Clipping
# costs 1.0 per SCORE_CLIP_SCALE percent of the ROI because saturated ball pixels give poor
# centroids, and a median above GOOD_MEDIAN_MAX costs 1.0 per SCORE_MEDIAN_SCALE gray levels.
# Exposure penalty (motion blur): costs SCORE_EXPOSURE_WEIGHT per millisecond of shutter time.
# At 2.0, trading 50 us of exposure for gain is worth about 10 points of lost ball contrast.
SCORE_EXPOSURE_WEIGHT = 2.0

# Gain above GAIN_COMFORT_LIMIT costs SCORE_GAIN_WEIGHT per unit (sensor noise). It is small
# enough that a clearly too-dim image still moves up, but a good image stays at low gain.
GAIN_COMFORT_LIMIT = 10.0
SCORE_GAIN_WEIGHT = 0.1
SCORE_PEAK_TARGET = 200.0
# Ball peak below GOOD_PEAK_MIN costs 1.0 per SCORE_PEAK_SHORTFALL_SCALE levels. A ball that cannot
# be detected makes blur avoidance moot, so this outweighs the exposure and gain penalties: the
# club and ball masks need that headroom, so a step that closes the gap is worth the extra blur or
# gain it costs. Once the ball clears GOOD_PEAK_MIN the term is zero and those penalties decide.
SCORE_PEAK_SHORTFALL_SCALE = 25.0
# Scene p90 below GOOD_SCENE_P90_MIN costs 1.0 per SCORE_SCENE_SHORTFALL_SCALE levels, for the same
# reason: a step that lights the club is worth its extra gain or blur.
SCORE_SCENE_SHORTFALL_SCALE = 20.0
SCORE_CONTRAST_TARGET = 100.0
SCORE_CLIP_SCALE = 2.0
SCORE_MEDIAN_SCALE = 50.0


@dataclass(frozen=True)
class ExposureStep:
    """One validated manual exposure/gain pair in increasing brightness order."""

    exposure_us: int
    gain: float
    label: str | None = None

    @property
    def signal(self) -> float:
        """Approximate relative image signal used for startup targeting."""
        return self.exposure_us * self.gain

    def to_dict(self) -> dict:
        """Serialize the step for API consumers."""
        return asdict(self)


# The ladder is ordered by image brightness (exposure x gain): the controller steps "brighter" and
# "darker" by index, so every entry must be at least as bright as the one before it (a test
# enforces this). Entries marked "new" are unvalidated pairs for very bright scenes: confirm
# them with calibrate_camera_exposure.py.
EXPOSURE_STEPS = (
    ExposureStep(60, 1.0),  # new: very bright conditions
    ExposureStep(80, 1.5),  # new: very bright conditions
    ExposureStep(100, 2.0),
    ExposureStep(100, 3.0),  # new: splits the 2.25x jump between 100/2.0 and 150/3.0
    ExposureStep(150, 3.0),
    ExposureStep(200, 3.0),
    ExposureStep(250, 4.0, "Outdoor sun"),
    ExposureStep(300, 5.0),
    ExposureStep(350, 6.0, "Outdoor shade"),
    ExposureStep(400, 8.0),
    ExposureStep(450, 10.0, "Evening"),
    ExposureStep(500, 12.0, "Indoor bright"),
    ExposureStep(500, 15.0),
    ExposureStep(650, 15.0, "Indoor dark"),
    ExposureStep(800, 18.0),
    ExposureStep(1000, 20.0, "Night"),
    ExposureStep(1250, 20.0, "Facility dark"),  # was 1250/16.0, the same brightness as 1000/20.0
)


@dataclass(frozen=True)
class ExposureObservation:
    """Brightness and contrast measurements from the impact-zone ROI."""

    sample_available: bool
    status: ExposureQualityStatus
    recommendation: ExposureRecommendation
    message: str
    p10: float | None = None
    median: float | None = None
    p90: float | None = None
    contrast: float | None = None
    clipped_pct: float | None = None
    dark_pct: float | None = None
    peak: float | None = None

    @property
    def acceptable(self) -> bool:
        """Whether camera-derived metrics may use frames with this exposure."""
        return self.status in {"good", "marginal"}

    @property
    def tolerable(self) -> bool:
        """Whether the frame is usable enough to accept when no better setting exists.

        Used only when the controller cannot improve the exposure any further.
        Rejects frames that are effectively black or almost entirely saturated.
        """
        if not self.sample_available:
            return False
        if self.status in {"good", "marginal"}:
            return True
        if self.status == "too_bright":
            contrast = self.ball_contrast
            return (
                (self.clipped_pct if self.clipped_pct is not None else 100.0)
                < TOLERABLE_MAX_CLIPPED_PCT
                and (self.median if self.median is not None else 255.0) < TOLERABLE_MAX_MEDIAN
                and (contrast is None or contrast >= MIN_BALL_CONTRAST)
            )
        if self.status == "too_dark":
            peak = self.peak if self.peak is not None else self.p90
            return peak is not None and peak >= TOLERABLE_MIN_PEAK
        return False

    @property
    def ball_contrast(self) -> float | None:
        """How far the ball (peak) stands above the background (median), when both are known."""
        if self.peak is None or self.median is None:
            return None
        return round(self.peak - self.median, 1)

    def to_dict(self) -> dict:
        """Serialize the observation for logs and the operator UI."""
        payload = asdict(self)
        payload["ball_contrast"] = self.ball_contrast
        return payload


@dataclass(frozen=True)
class AutoExposureDecision:
    """One side-effect-free controller decision."""

    status: AutoExposureStatus
    analysis_eligible: bool
    message: str
    observation: ExposureObservation
    target: ExposureStep | None = None
    motion_blur_risk: Literal["low", "elevated", "high"] = "low"

    @property
    def should_apply(self) -> bool:
        """Whether the runtime should apply a new camera setting."""
        return self.target is not None

    def to_dict(self) -> dict:
        """Serialize the decision for status and logging."""
        payload = {
            "status": self.status,
            "analysis_eligible": self.analysis_eligible,
            "message": self.message,
            "motion_blur_risk": self.motion_blur_risk,
            "observation": self.observation.to_dict(),
        }
        payload["target"] = self.target.to_dict() if self.target else None
        return payload


def exposure_steps_for_fps(fps: float) -> tuple[ExposureStep, ...]:
    """Return ladder entries that fit inside the requested frame period."""
    if fps <= 0:
        raise ValueError("camera FPS must be positive")
    frame_period_us = round(1_000_000 / fps)
    steps = tuple(step for step in EXPOSURE_STEPS if step.exposure_us < frame_period_us)
    if not steps:
        raise ValueError(f"no exposure steps fit inside the {frame_period_us}us frame period")
    return steps


def measure_exposure(image: np.ndarray) -> ExposureObservation:
    """Rate exposure in the center-lower hitting zone of one grayscale frame."""
    pixels = np.asarray(image)
    if pixels.ndim != 2 or not pixels.size:
        return ExposureObservation(
            sample_available=False,
            status="unavailable",
            recommendation="hold",
            message="Waiting for a camera frame",
        )

    height, width = pixels.shape
    region = pixels[
        round(height * 0.45) : round(height * 0.9),
        round(width * 0.2) : round(width * 0.8),
    ]
    if not region.size:
        return ExposureObservation(
            sample_available=False,
            status="unavailable",
            recommendation="hold",
            message="Impact-area exposure region is unavailable",
        )

    p10, median, p90, peak = (
        float(np.percentile(region, value)) for value in (10, 50, 90, BRIGHT_FEATURE_PERCENTILE)
    )
    clipped_pct = float(np.mean(region >= 250) * 100.0)
    dark_pct = float(np.mean(region <= 12) * 100.0)
    contrast = p90 - p10
    ball_contrast = peak - median
    washed_out = median >= WASHED_OUT_MEDIAN and ball_contrast < MIN_BALL_CONTRAST

    if clipped_pct >= TOO_BRIGHT_CLIPPED_PCT or median >= TOO_BRIGHT_MEDIAN or washed_out:
        status: ExposureQualityStatus = "too_bright"
        recommendation: ExposureRecommendation = "darker"
        message = (
            "Impact area is clipping; reducing exposure"
            if clipped_pct >= TOO_BRIGHT_CLIPPED_PCT or median >= TOO_BRIGHT_MEDIAN
            else "Ball is washed out against a bright background; reducing exposure"
        )
    elif peak < TOO_DARK_PEAK:
        # Nothing bright enough to image the ball or clubhead. A dark background by itself
        # (low median / low contrast) is fine and no longer counts against the image.
        status = "too_dark"
        recommendation = "brighter"
        if peak <= NEARLY_BLACK_PEAK and dark_pct >= 90.0:
            message = "Camera view is nearly black; check lighting and the lens cover"
        else:
            message = "Ball and club are faint; increasing exposure"
    elif (
        clipped_pct <= GOOD_CLIPPED_MAX_PCT
        and median <= GOOD_MEDIAN_MAX
        and peak >= GOOD_PEAK_MIN
        and p90 >= GOOD_SCENE_P90_MIN
        and ball_contrast >= GOOD_MIN_CONTRAST
    ):
        status = "good"
        recommendation = "hold"
        message = "Impact-area exposure and contrast look good"
    else:
        status = "marginal"
        too_bright_side = (
            clipped_pct > GOOD_CLIPPED_MAX_PCT
            or median > GOOD_MEDIAN_MAX
            or (median >= WASHED_OUT_MEDIAN and ball_contrast < GOOD_MIN_CONTRAST)
        )
        recommendation = "darker" if too_bright_side else "brighter"
        message = f"Exposure is usable; a {recommendation} setting may improve contrast"

    return ExposureObservation(
        sample_available=True,
        status=status,
        recommendation=recommendation,
        message=message,
        p10=round(p10, 1),
        median=round(median, 1),
        p90=round(p90, 1),
        contrast=round(contrast, 1),
        clipped_pct=round(clipped_pct, 2),
        dark_pct=round(dark_pct, 2),
        peak=round(peak, 1),
    )


def motion_blur_risk(exposure_us: int) -> Literal["low", "elevated", "high"]:
    """Describe the clubhead blur risk introduced by shutter duration."""
    if exposure_us > 800:
        return "high"
    if exposure_us > 500:
        return "elevated"
    return "low"


def exposure_quality_score(
    observation: ExposureObservation,
    exposure_us: int | None = None,
    gain: float | None = None,
) -> float:
    """Rank frames so refinement can tell whether a step helped (higher is better).

    Shorter exposure scores higher when exposure_us is given (blur on the fast clubhead and
    ball), and gain above GAIN_COMFORT_LIMIT scores lower when gain is given (sensor noise).
    """
    if not observation.sample_available:
        return float("-inf")
    peak = observation.peak if observation.peak is not None else (observation.p90 or 0.0)
    clipped = observation.clipped_pct or 0.0
    median = observation.median or 0.0
    scene = observation.p90 if observation.p90 is not None else GOOD_SCENE_P90_MIN
    blur_penalty = SCORE_EXPOSURE_WEIGHT * (exposure_us or 0) / 1000.0
    noise_penalty = SCORE_GAIN_WEIGHT * max(0.0, (gain or 0.0) - GAIN_COMFORT_LIMIT)
    return (
        min(peak, SCORE_PEAK_TARGET) / SCORE_PEAK_TARGET
        + min(max(peak - median, 0.0), SCORE_CONTRAST_TARGET) / SCORE_CONTRAST_TARGET
        - clipped / SCORE_CLIP_SCALE
        - max(0.0, median - GOOD_MEDIAN_MAX) / SCORE_MEDIAN_SCALE
        - max(0.0, GOOD_PEAK_MIN - peak) / SCORE_PEAK_SHORTFALL_SCALE
        - max(0.0, GOOD_SCENE_P90_MIN - scene) / SCORE_SCENE_SHORTFALL_SCALE
        - blur_penalty
        - noise_penalty
    )


class AutoExposurePolicy:
    """Choose startup exposure steps, refine marginal results, and lock the final decision."""

    def __init__(
        self,
        *,
        fps: float,
        startup_max_adjustments: int = 3,
        refine_max_steps: int = REFINE_MAX_STEPS,
        probe_good_neighbors: bool = True,
    ):
        if startup_max_adjustments < 1:
            raise ValueError("startup_max_adjustments must be positive")
        if refine_max_steps < 0:
            raise ValueError("refine_max_steps must not be negative")
        self.steps = exposure_steps_for_fps(fps)
        self.startup_max_adjustments = startup_max_adjustments
        self.refine_max_steps = refine_max_steps
        self.probe_good_neighbors = probe_good_neighbors
        self._startup = True
        self._startup_adjustments = 0
        self._locked_decision: AutoExposureDecision | None = None
        self._refining = False
        self._refine_steps = 0
        self._refine_direction = 0
        self._probe_queue: list[int] = []
        self._climb_from_good = False
        self._best: tuple[int, ExposureObservation, float] | None = None
        self._final_index: int | None = None

    def _reset_startup_state(self) -> None:
        """Enter startup convergence and clear any prior locked result."""
        self._startup = True
        self._startup_adjustments = 0
        self._locked_decision = None
        self._refining = False
        self._refine_steps = 0
        self._refine_direction = 0
        self._probe_queue = []
        self._climb_from_good = False
        self._best = None
        self._final_index = None

    def reset(self) -> None:
        """Unlock the policy for a new camera startup."""
        self._reset_startup_state()

    @property
    def startup(self) -> bool:
        """Whether the policy is still using fast startup convergence."""
        return self._startup

    def evaluate(
        self,
        observation: ExposureObservation,
        *,
        exposure_us: int,
        gain: float,
    ) -> AutoExposureDecision:
        """Return the next camera-control decision without applying it."""
        if self._locked_decision is not None:
            return self._locked_decision

        current_index = self._nearest_step_index(exposure_us, gain)
        current_step = self.steps[current_index]
        risk = motion_blur_risk(current_step.exposure_us)

        if not observation.sample_available:
            return AutoExposureDecision(
                status="unavailable",
                analysis_eligible=False,
                message=observation.message,
                observation=observation,
                motion_blur_risk=risk,
            )

        if self._final_index is not None:
            return self._restore_best_decision(observation, current_index, risk)

        if self._refining:
            return self._refine_decision(observation, current_index)

        if observation.acceptable:
            return self._first_acceptable_decision(observation, current_index, risk)

        return self._startup_decision(observation, current_index)

    def _lock_ready(
        self,
        observation: ExposureObservation,
        risk: Literal["low", "elevated", "high"],
    ) -> AutoExposureDecision:
        """Lock the final usable setting."""
        self._startup = False
        self._startup_adjustments = 0
        self._refining = False
        self._probe_queue = []
        self._final_index = None
        self._locked_decision = AutoExposureDecision(
            status="ready",
            analysis_eligible=True,
            message=observation.message,
            observation=observation,
            motion_blur_risk=risk,
        )
        return self._locked_decision

    def _adjust_decision(
        self,
        observation: ExposureObservation,
        target_index: int,
        message: str,
    ) -> AutoExposureDecision:
        target = self.steps[target_index]
        return AutoExposureDecision(
            status="adjusting",
            analysis_eligible=False,
            message=message,
            observation=observation,
            target=target,
            motion_blur_risk=motion_blur_risk(target.exposure_us),
        )

    def _score(self, observation: ExposureObservation, step_index: int) -> float:
        step = self.steps[step_index]
        return exposure_quality_score(observation, step.exposure_us, step.gain)

    def _refine_message(self) -> str:
        return f"Refining camera exposure ({self._refine_steps}/{self.refine_max_steps})"

    def _first_acceptable_decision(
        self,
        observation: ExposureObservation,
        current_index: int,
        risk: Literal["low", "elevated", "high"],
    ) -> AutoExposureDecision:
        """Lock when refinement is off; otherwise start probing neighboring steps.

        A marginal frame probes one step in the recommended direction. A good frame probes one
        step in each direction (darker first, which also means less blur and noise) to confirm
        that no neighbor is clearly better.
        """
        if self.refine_max_steps == 0:
            return self._lock_ready(observation, risk)

        from_good = observation.status == "good"
        if from_good:
            if not self.probe_good_neighbors:
                return self._lock_ready(observation, risk)
            candidates: tuple[int, ...] = (-1, 1)
        else:
            candidates = ({"brighter": 1, "darker": -1}.get(observation.recommendation, 0),)
        directions = [
            step for step in candidates if step != 0 and 0 <= current_index + step < len(self.steps)
        ]
        if not directions:
            return self._lock_ready(observation, risk)

        # Stay in fast-startup mode (self._startup stays True) until refinement finishes.
        self._startup_adjustments = 0
        self._refining = True
        self._climb_from_good = from_good
        self._best = (
            current_index,
            observation,
            self._score(observation, current_index),
        )
        self._refine_direction = directions[0]
        self._probe_queue = directions[1:]
        self._refine_steps = 1
        return self._adjust_decision(
            observation, current_index + directions[0], self._refine_message()
        )

    def _refine_decision(
        self,
        observation: ExposureObservation,
        current_index: int,
    ) -> AutoExposureDecision:
        """Climb while the score improves, probe the other side once, else keep the best step."""
        assert self._best is not None
        risk = motion_blur_risk(self.steps[current_index].exposure_us)
        best_index, _, best_score = self._best
        score = self._score(observation, current_index)
        upgraded = observation.status == "good" and self._best[1].status != "good"
        improved = observation.acceptable and (
            upgraded or score > best_score + REFINE_MIN_IMPROVEMENT
        )

        if improved:
            self._best = (current_index, observation, score)
            self._probe_queue = []  # this direction helps, so the opposite side is not needed
            next_index = current_index + self._refine_direction
            # A marginal start stops at the first "good" frame. The exception is a darker climb
            # that is still above the gain comfort limit or has elevated blur risk: a
            # lower-gain or shorter-exposure neighbor may be good too.
            step = self.steps[current_index]
            good_enough = (
                observation.status == "good"
                and not self._climb_from_good
                and (
                    self._refine_direction > 0
                    or (step.gain <= GAIN_COMFORT_LIMIT and risk == "low")
                )
            )
            if (
                good_enough
                or self._refine_steps >= self.refine_max_steps
                or not 0 <= next_index < len(self.steps)
            ):
                return self._lock_ready(observation, risk)
            self._refine_steps += 1
            return self._adjust_decision(observation, next_index, self._refine_message())

        if self._probe_queue and self._refine_steps < self.refine_max_steps:
            # Nothing better on this side of the starting step: jump straight to the other side.
            self._refine_direction = self._probe_queue.pop(0)
            self._refine_steps += 1
            return self._adjust_decision(
                observation, best_index + self._refine_direction, self._refine_message()
            )

        return self._settle_on_best(observation, current_index)

    def _settle_on_best(
        self,
        observation: ExposureObservation,
        current_index: int,
    ) -> AutoExposureDecision:
        """Stop refining; return to the best step if the camera is no longer on it."""
        assert self._best is not None
        best_index, best_observation, _ = self._best
        self._refining = False
        if best_index == current_index:
            return self._lock_ready(
                best_observation, motion_blur_risk(self.steps[best_index].exposure_us)
            )
        self._final_index = best_index
        return self._adjust_decision(observation, best_index, "Restoring best camera exposure")

    def _restore_best_decision(
        self,
        observation: ExposureObservation,
        current_index: int,
        risk: Literal["low", "elevated", "high"],
    ) -> AutoExposureDecision:
        """Lock once the camera is back on the best step found during refinement."""
        assert self._best is not None and self._final_index is not None
        if current_index != self._final_index:
            return self._adjust_decision(
                observation, self._final_index, "Restoring best camera exposure"
            )
        chosen = observation if observation.acceptable else self._best[1]
        return self._lock_ready(chosen, risk)

    def _startup_decision(
        self,
        observation: ExposureObservation,
        current_index: int,
    ) -> AutoExposureDecision:
        current = self.steps[current_index]
        risk = motion_blur_risk(current.exposure_us)
        if self._startup_adjustments >= self.startup_max_adjustments:
            return self._settle_without_convergence(observation, risk)

        target_index = self._startup_target_index(observation, current_index)
        if target_index == current_index:
            return self._settle_without_convergence(observation, risk)

        self._startup_adjustments += 1
        target = self.steps[target_index]
        return AutoExposureDecision(
            status="adjusting",
            analysis_eligible=False,
            message=(
                "Calibrating camera exposure "
                f"({self._startup_adjustments}/{self.startup_max_adjustments})"
            ),
            observation=observation,
            target=target,
            motion_blur_risk=motion_blur_risk(target.exposure_us),
        )

    def _settle_without_convergence(
        self,
        observation: ExposureObservation,
        risk: Literal["low", "elevated", "high"],
    ) -> AutoExposureDecision:
        """Lock a result when no better setting is reachable.

        A frame that is imperfect but not catastrophic is accepted (analysis stays quality-gated
        downstream). Only near-black or almost fully saturated frames report lighting_required.
        """
        self._startup = False
        if observation.tolerable:
            self._locked_decision = AutoExposureDecision(
                status="ready",
                analysis_eligible=True,
                message="Best available exposure selected; image quality may be reduced",
                observation=observation,
                motion_blur_risk=risk,
            )
        else:
            self._locked_decision = AutoExposureDecision(
                status="lighting_required",
                analysis_eligible=False,
                message=self._lighting_message(observation),
                observation=observation,
                motion_blur_risk=risk,
            )
        return self._locked_decision

    def _startup_target_index(
        self,
        observation: ExposureObservation,
        current_index: int,
    ) -> int:
        current = self.steps[current_index]
        if observation.recommendation == "brighter":
            # Dark backgrounds are fine, so scale by the bright feature (ball/club), not the median.
            peak = observation.peak if observation.peak is not None else observation.median
            scale = BRIGHTER_TARGET_PEAK / max(float(peak or 1.0), 1.0)
        else:
            median = max(float(observation.median or 1.0), 1.0)
            scale = DARKER_TARGET_MEDIAN / median
        target_signal = current.signal * max(0.125, min(8.0, scale))
        target_index = min(
            range(len(self.steps)),
            key=lambda index: abs(math.log(self.steps[index].signal / target_signal)),
        )
        if observation.recommendation == "brighter":
            return (
                max(current_index + 1, target_index)
                if current_index < len(self.steps) - 1
                else current_index
            )
        if observation.recommendation == "darker":
            return min(current_index - 1, target_index) if current_index > 0 else current_index
        return current_index

    def _nearest_step_index(self, exposure_us: int, gain: float) -> int:
        exact_index = next(
            (
                index
                for index, step in enumerate(self.steps)
                if step.exposure_us == exposure_us and math.isclose(step.gain, gain)
            ),
            None,
        )
        if exact_index is not None:
            return exact_index
        signal = max(float(exposure_us) * float(gain), 1.0)
        return min(
            range(len(self.steps)),
            key=lambda index: abs(math.log(self.steps[index].signal / signal)),
        )

    @staticmethod
    def _lighting_message(observation: ExposureObservation) -> str:
        if observation.status == "too_dark":
            return "Camera lighting is insufficient; add or redirect light toward the ball"
        return "Camera view is overexposed; reduce or redirect light near the ball"
