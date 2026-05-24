from __future__ import annotations

import inspect
from typing import Any, Callable


def accepted_call_parameters(callable_owner: Any) -> set[str]:
    try:
        return set(inspect.signature(callable_owner.__call__).parameters)
    except (TypeError, ValueError):
        return set()


def filter_call_kwargs(callable_owner: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    accepted = accepted_call_parameters(callable_owner)
    if not accepted:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in accepted}


def cancel_step_callback(
    pipe: Any,
    cancel_requested: Callable[[], bool] | None,
) -> Callable[[Any, int, Any, dict[str, Any]], dict[str, Any]] | None:
    """Build a diffusers ``callback_on_step_end`` that interrupts the denoise
    loop the moment a job is canceled.

    The callback flips ``pipe._interrupt`` — diffusers and the vendored Lens
    pipeline check that flag at the top of each step and break cleanly, so a
    cancel lands at the next step boundary instead of after the whole run.
    Returns ``None`` when there's nothing to wire (no cancel predicate, or the
    pipe's ``__call__`` doesn't accept ``callback_on_step_end``); the caller then
    just runs the pipe normally and relies on the worker's cancel watchdog.

    Note: pipelines that *accept* the callback but ignore ``_interrupt`` will run
    to completion anyway, so callers MUST re-check cancellation after the pipe
    returns and discard the (possibly partial) result."""
    if cancel_requested is None:
        return None
    if "callback_on_step_end" not in accepted_call_parameters(pipe):
        return None

    def _on_step_end(
        active_pipe: Any,
        step: int,
        timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if cancel_requested():
            active_pipe._interrupt = True
        return callback_kwargs

    return _on_step_end
