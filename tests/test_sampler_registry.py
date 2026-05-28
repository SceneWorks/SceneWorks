"""Tests for scene_worker.sampler_registry (epic 1753).

The registry must be self-contained — no real diffusers import. We use a
fake `diffusers` module with stub scheduler classes that mirror the public
shape: a `config` mapping, a `from_config(config, **overrides)` classmethod,
and an `__init__` signature whose accepted params govern feature detection.
"""

from __future__ import annotations

import inspect
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from scene_worker.sampler_registry import (
    SUPPORTED_SAMPLERS,
    SUPPORTED_SCHEDULERS,
    apply_sampler,
    sampler_selection_from_advanced,
)


class _StubScheduler:
    """Base class for stub schedulers. Subclasses declare their accepted
    config keys via their ``__init__`` signature so the registry's feature
    detection treats them like a real diffusers scheduler."""

    def __init__(self, *, num_train_timesteps: int = 1000, **kwargs: Any) -> None:
        self.config = SimpleNamespace(
            num_train_timesteps=num_train_timesteps, **kwargs
        )

    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> "_StubScheduler":
        # Mirror diffusers ConfigMixin semantics: extract only the params the
        # target class accepts (drops irrelevant flags from a different
        # scheduler's config), then layer overrides.
        if hasattr(config, "items"):
            base = dict(config.items())
        elif hasattr(config, "__dict__"):
            base = {
                key: value
                for key, value in vars(config).items()
                if not key.startswith("_")
            }
        else:
            base = {}
        accepted = set(inspect.signature(cls.__init__).parameters)
        merged = {key: value for key, value in base.items() if key in accepted}
        for key, value in overrides.items():
            if key in accepted:
                merged[key] = value
        return cls(**merged)


class FlowMatchEulerDiscreteScheduler(_StubScheduler):
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_dynamic_shifting: bool = False,
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
    ) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
        )


class FlowMatchHeunDiscreteScheduler(_StubScheduler):
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_dynamic_shifting: bool = False,
    ) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
        )


class DPMSolverMultistepScheduler(_StubScheduler):
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        use_flow_sigmas: bool = False,
        prediction_type: str = "epsilon",
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
    ) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            use_flow_sigmas=use_flow_sigmas,
            prediction_type=prediction_type,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
        )


class UniPCMultistepScheduler(_StubScheduler):
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        use_flow_sigmas: bool = False,
        use_karras_sigmas: bool = False,
    ) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            use_flow_sigmas=use_flow_sigmas,
            use_karras_sigmas=use_karras_sigmas,
        )


class OldHeunScheduler(_StubScheduler):
    """Simulates an older diffusers build where FlowMatchHeun has no
    ``use_dynamic_shifting`` parameter — registry must drop the flag."""

    def __init__(self, *, num_train_timesteps: int = 1000, shift: float = 3.0) -> None:
        super().__init__(num_train_timesteps=num_train_timesteps, shift=shift)


@pytest.fixture
def fake_diffusers(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install a fake ``diffusers`` module with stub scheduler classes."""
    module = ModuleType("diffusers")
    module.FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler
    module.FlowMatchHeunDiscreteScheduler = FlowMatchHeunDiscreteScheduler
    module.DPMSolverMultistepScheduler = DPMSolverMultistepScheduler
    module.UniPCMultistepScheduler = UniPCMultistepScheduler
    monkeypatch.setitem(sys.modules, "diffusers", module)
    return module


def _make_pipe(scheduler: Any) -> SimpleNamespace:
    return SimpleNamespace(scheduler=scheduler)


def _make_default_pipe() -> SimpleNamespace:
    # Mirror a real diffusers FlowMatch pipe: scheduler has a `config` view
    # that exposes the trained params.
    base = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    return _make_pipe(base)


def test_supported_sets_are_documented() -> None:
    assert "default" in SUPPORTED_SAMPLERS
    assert set(SUPPORTED_SAMPLERS) >= {"default", "euler", "heun", "dpmpp", "unipc"}
    assert set(SUPPORTED_SCHEDULERS) >= {
        "default",
        "simple",
        "shift",
        "karras",
        "exponential",
        "beta",
    }


def test_default_default_is_a_noop(fake_diffusers: ModuleType) -> None:
    pipe = _make_default_pipe()
    original = pipe.scheduler
    result = apply_sampler(pipe, "default", "default")
    assert result["noop"] is True
    assert pipe.scheduler is original


def test_unknown_sampler_falls_back_without_mutation(
    fake_diffusers: ModuleType,
) -> None:
    pipe = _make_default_pipe()
    original = pipe.scheduler
    result = apply_sampler(pipe, "lcm", "default")
    assert result["noop"] is True
    assert pipe.scheduler is original


def test_unknown_scheduler_falls_back_without_mutation(
    fake_diffusers: ModuleType,
) -> None:
    pipe = _make_default_pipe()
    original = pipe.scheduler
    result = apply_sampler(pipe, "default", "lms")
    assert result["noop"] is True
    assert pipe.scheduler is original


def test_each_supported_sampler_constructs_a_valid_scheduler(
    fake_diffusers: ModuleType,
) -> None:
    for sampler in ("euler", "heun", "dpmpp", "unipc"):
        pipe = _make_default_pipe()
        result = apply_sampler(pipe, sampler, "default")
        assert result["noop"] is False
        assert result["sampler"] == sampler
        assert pipe.scheduler is not None


def test_dpmpp_pins_flow_mode_overrides(fake_diffusers: ModuleType) -> None:
    pipe = _make_default_pipe()
    apply_sampler(pipe, "dpmpp", "default")
    assert isinstance(pipe.scheduler, DPMSolverMultistepScheduler)
    assert pipe.scheduler.config.use_flow_sigmas is True
    assert pipe.scheduler.config.prediction_type == "flow_prediction"


def test_unipc_pins_flow_sigmas(fake_diffusers: ModuleType) -> None:
    pipe = _make_default_pipe()
    apply_sampler(pipe, "unipc", "default")
    assert isinstance(pipe.scheduler, UniPCMultistepScheduler)
    assert pipe.scheduler.config.use_flow_sigmas is True


def test_karras_sigma_flag_is_threaded_when_supported(
    fake_diffusers: ModuleType,
) -> None:
    pipe = _make_default_pipe()
    apply_sampler(pipe, "dpmpp", "karras")
    assert pipe.scheduler.config.use_karras_sigmas is True
    assert pipe.scheduler.config.use_flow_sigmas is True


def test_beta_sigma_flag_is_threaded_when_supported(
    fake_diffusers: ModuleType,
) -> None:
    pipe = _make_default_pipe()
    apply_sampler(pipe, "euler", "beta")
    assert pipe.scheduler.config.use_beta_sigmas is True


def test_simple_explicitly_clears_alt_sigma_flags(
    fake_diffusers: ModuleType,
) -> None:
    # Start with a Karras config bleed-through, then ask for "simple" —
    # the alt flags must be turned back off explicitly.
    sticky = DPMSolverMultistepScheduler(
        use_flow_sigmas=True,
        prediction_type="flow_prediction",
        use_karras_sigmas=True,
    )
    pipe = _make_pipe(sticky)
    apply_sampler(pipe, "dpmpp", "simple")
    assert pipe.scheduler.config.use_karras_sigmas is False
    assert pipe.scheduler.config.use_exponential_sigmas is False
    assert pipe.scheduler.config.use_beta_sigmas is False


def test_shift_applies_value_and_disables_dynamic_shifting(
    fake_diffusers: ModuleType,
) -> None:
    pipe = _make_default_pipe()
    result = apply_sampler(pipe, "euler", "shift", 5.5)
    assert result["shift"] == pytest.approx(5.5)
    assert pipe.scheduler.config.shift == pytest.approx(5.5)
    assert pipe.scheduler.config.use_dynamic_shifting is False


def test_shift_value_is_ignored_when_scheduler_is_not_shift(
    fake_diffusers: ModuleType,
) -> None:
    pipe = _make_default_pipe()
    apply_sampler(pipe, "euler", "karras", 7.0)
    # No `shift` field expected to be set explicitly from the caller axis
    # (it carries through whatever the base config had — 3.0 here).
    assert pipe.scheduler.config.shift == pytest.approx(3.0)


def test_invalid_shift_falls_back_to_default(fake_diffusers: ModuleType) -> None:
    pipe = _make_default_pipe()
    result = apply_sampler(pipe, "euler", "shift", "nope")
    # Selection still applied, just without the shift override.
    assert result["shift"] is None


def test_feature_detection_drops_unsupported_flags(
    fake_diffusers: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Swap Heun for the old-stub that lacks `use_dynamic_shifting`.
    fake_diffusers.FlowMatchHeunDiscreteScheduler = OldHeunScheduler  # type: ignore[attr-defined]
    pipe = _make_default_pipe()
    result = apply_sampler(pipe, "heun", "shift", 4.0)
    assert "use_dynamic_shifting" in result["droppedFlags"]
    # Shift still landed.
    assert pipe.scheduler.config.shift == pytest.approx(4.0)


def test_missing_scheduler_class_falls_back(
    fake_diffusers: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Remove a class to simulate diffusers without it.
    monkeypatch.delattr(fake_diffusers, "UniPCMultistepScheduler", raising=False)
    pipe = _make_default_pipe()
    original = pipe.scheduler
    result = apply_sampler(pipe, "unipc", "default")
    assert result["noop"] is True
    assert pipe.scheduler is original


def test_default_restores_original_after_non_default_swap(
    fake_diffusers: ModuleType,
) -> None:
    pipe = _make_default_pipe()
    original_cls = type(pipe.scheduler)
    apply_sampler(pipe, "dpmpp", "karras")
    assert isinstance(pipe.scheduler, DPMSolverMultistepScheduler)
    result = apply_sampler(pipe, "default", "default")
    assert result["restored"] is True
    # Restored class matches the original FlowMatchEuler.
    assert isinstance(pipe.scheduler, original_cls)


def test_no_scheduler_attribute_is_treated_as_noop(
    fake_diffusers: ModuleType,
) -> None:
    pipe = SimpleNamespace()  # no `scheduler` attr at all
    result = apply_sampler(pipe, "euler", "karras")
    assert result["noop"] is True


def test_sampler_selection_from_advanced_normalizes_inputs() -> None:
    assert sampler_selection_from_advanced({}) == ("default", "default", None)
    assert sampler_selection_from_advanced(
        {"sampler": "EULER", "scheduler": "Karras"}
    ) == ("euler", "karras", None)
    assert sampler_selection_from_advanced(
        {"sampler": "garbage", "scheduler": "garbage"}
    ) == ("default", "default", None)
    assert sampler_selection_from_advanced(
        {"sampler": "euler", "scheduler": "shift", "schedulerShift": "4.5"}
    ) == ("euler", "shift", pytest.approx(4.5))
    # Shift value is dropped when scheduler != "shift".
    assert sampler_selection_from_advanced(
        {"sampler": "euler", "scheduler": "karras", "schedulerShift": "4.5"}
    ) == ("euler", "karras", None)
    # Bad payload type.
    assert sampler_selection_from_advanced(None) == ("default", "default", None)  # type: ignore[arg-type]
