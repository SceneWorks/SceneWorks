"""Print one compact worker row from a verified SCAIL physical capture and its review.

Run after the inference producer's --verify-artifacts has passed. The review JSON
names the immutable artifact_id, run_id, run_attempt, receipt_sha256, references, rationale, and
one entry per domain in "domains". Each entry supplies a "basis" and only its
reviewed extra/uncertainty or overshoot bytes. Capture facts come from the files.
A CUDA measured-aggregate allowance must cover the whole allocation burst (see
burst_growth); an MLX price must reach the native decode budget.

    python scripts/project-scail2-memory.py path/to/receipt.json physical-review.json

For the archived GPU0 six, supply --recovery-archive and --recovery-certificate.
The independent review must name the hash of the native verifier's certificate.
The original campaign acceptance remains false. Device observations end at the
completed RGB output. The separate Windows Job counters cover host lifetime.
For the four-count GPU0 continuation, supply the original --recovery-archive for
each count. Counts 3 and 5 also require the native owned-zero certificate map.
The review names the SHA-256 of the count's sorted compact JSON certificate plus
one newline. Each projection retains the original campaign and count verdicts.
For a fresh full plan with supervised Job completion, the review supplies
hosted_plan_sha256 from the original hosted plan. No certificate is needed. Such a plan
may name GPU0 or GPU1 or name no card; a card-free receipt records its most-free card
under cuda_card_selection. Either card qualifies on the same terms, bound to the card
UUID and physical index the receipt recorded.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


# The Windows CUDA host's two identical RTX PRO 6000 Blackwell Max-Q cards (97,887 MiB each):
# UUID -> nvidia-smi physical index. GPU1 also drives the display. A SCAIL owned-Job capture may
# run on either card; projection binds the card UUID and index the receipt recorded.
GPU0 = "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0"
GPU0_PCI = "00000000:21:00.0"
REGISTERED_CUDA_CARDS = {GPU0: 0, "GPU-e4b79931-7be6-f216-460a-f5405cfafffe": 1}
# Inference's card-free campaign records its most-free-card choice under this rule.
CARD_SELECTION_RULE = "most-free-card-covering-estimate-plus-reserve-v1"
CARD_SELECTION_ESTIMATE_SOURCE = "scail2-plural-reviewed-device-peak-v1"
CARD_SELECTION_ESTIMATE_BYTES = 60 << 30
# The most an idle card may hold before admission and still count as quiet at release. Recorded
# idle GPU0 admission samples hold 19 MiB (and 498 MiB in run 45265278792), so driver contexts
# get 512 MiB. GPU1 also drives the desktop: its recorded baseline is 908 MiB
# (docs/testing/sc-26235) under a 2,048 MiB prelaunch guard, so it gets a further 1,536 MiB.
# Admission samples must agree within 64 MiB at 0% utilization.
DRIVER_CONTEXT_IDLE_MIB = 512
DISPLAY_IDLE_ALLOWANCE_MIB = {"GPU-e4b79931-7be6-f216-460a-f5405cfafffe": 1536}
IDLE_SPREAD_MIB = 64


def card_total_mib(hw):
    total = str(hw.get("memory.total", "")).strip()
    require(re.fullmatch(r"[1-9][0-9]*", total), "CUDA capture lacks the card's total memory")
    return int(total)


def registered_card_index(uuid):
    require(uuid in REGISTERED_CUDA_CARDS, "CUDA capture ran on an unregistered card")
    return REGISTERED_CUDA_CARDS[uuid]


def cards_read(record, policy, expected):
    """The card one selection record chose, after checking it read exactly `expected` and chose
    the most-free card that fit. Mirrors inference's `_cards_read` (SceneWorks/inference#1509)."""
    cards = record.get("cards") if isinstance(record, dict) else None
    uuids = [card.get("uuid") for card in cards] if isinstance(cards, list) else None
    chosen = record.get("chosen_uuid") if isinstance(record, dict) else None
    reserve = policy.get("gpuFreeReserveBytes")
    require(
        isinstance(record, dict)
        and record.get("rule") == CARD_SELECTION_RULE
        and uuids == sorted(expected, key=REGISTERED_CUDA_CARDS.get)
        and chosen in uuids
        and record.get("chosen_index") == REGISTERED_CUDA_CARDS[chosen]
        and all(
            card.get("index") == REGISTERED_CUDA_CARDS[card["uuid"]]
            and type(card.get("estimate_bytes")) is int
            and 0 < card["estimate_bytes"] <= policy.get("childGpuCapBytes", 0)
            and card.get("required_free_bytes") == reserve + card["estimate_bytes"]
            for card in cards
        ),
        "card-free campaign lacks its recorded registered card selection",
    )
    fitting = [card for card in cards if "refusal" not in card]
    require(
        fitting
        and all(
            type(card.get("free_mib")) is int
            and card["free_mib"] << 20 >= card["required_free_bytes"]
            for card in fitting
        )
        and chosen == max(fitting, key=lambda card: (card["free_mib"], -card["index"]))["uuid"],
        "card-free campaign did not run on its recorded most-free card",
    )
    return chosen


def validate_card_selection(receipt, record):
    """A card-free campaign must have run on the card its final recorded selection chose.

    Mirrors inference's `executed_card_policy`: the first selection read every registered card;
    each switch (`superseded`) records why it left the card the previous selection chose, and the
    next selection read only the other card(s). Every selection used the SCAIL device estimate."""
    rule = receipt["contract"].get("cuda_card_selection")
    selection = receipt.get("cuda_card_selection")
    if rule is None:
        require(selection is None, "a named-card campaign records no card selection")
        return
    policy = receipt["contract"]["request"]["policy"]
    require(
        rule == CARD_SELECTION_RULE and "cudaDeviceUuid" not in policy,
        "card-free campaign lacks its recorded registered card selection",
    )
    superseded = selection.get("superseded", []) if isinstance(selection, dict) else None
    require(isinstance(superseded, list), "card-free campaign records an invalid card re-selection")
    previous, expected = None, list(REGISTERED_CUDA_CARDS)
    for earlier in superseded:
        require(
            isinstance(earlier, dict)
            and isinstance(earlier.get("reason"), str)
            and earlier["reason"]
            and cards_read(earlier.get("selection"), policy, expected) == earlier.get("left_uuid")
            and earlier["left_uuid"] != earlier.get("switched_to"),
            "card-free campaign records an invalid card re-selection",
        )
        previous = earlier
        expected = [uuid for uuid in REGISTERED_CUDA_CARDS if uuid != earlier["left_uuid"]]
    chosen = cards_read(selection, policy, expected)
    hw = receipt["hardware"]
    every = [selection, *(earlier["selection"] for earlier in superseded)]
    require(
        (previous is None or previous["switched_to"] == chosen)
        and all(
            (card.get("estimate_source"), card["estimate_bytes"])
            == (CARD_SELECTION_ESTIMATE_SOURCE, CARD_SELECTION_ESTIMATE_BYTES)
            for record in every
            for card in record["cards"]
        )
        and hw["uuid"] == chosen
        and str(hw["index"]) == str(REGISTERED_CUDA_CARDS[chosen]),
        "card-free campaign did not run on its recorded most-free card",
    )
    # The projected count's claim holds that card and states the SCAIL device estimate.
    claim = record["claim"]
    stated = claim.get("claim_gpu_estimate")
    require(
        claim.get("device_uuid") == chosen
        and isinstance(stated, dict)
        and (stated.get("source"), stated.get("estimate_bytes"))
        == (CARD_SELECTION_ESTIMATE_SOURCE, CARD_SELECTION_ESTIMATE_BYTES),
        "card-free claim lacks its card or stated device estimate",
    )


def executed_policy(receipt):
    """The signed request policy, or for a card-free request the policy of the recorded card."""
    policy = receipt["contract"]["request"]["policy"]
    if "cuda_card_selection" in receipt["contract"]:
        require("cudaDeviceUuid" not in policy, "card-free request names a card")
        return dict(policy, cudaDeviceUuid=receipt["hardware"]["uuid"])
    return policy


def byte_count(value):
    require(type(value) is int and 0 <= value < 2**64, "invalid byte count")
    return value


def burst_growth(values):
    """Growth across the continuous rising run that holds the largest single-poll step.

    One allocation can land across two or more polls (37.69 -> 46.07 -> 52.47 GiB), so a
    single step understates it. The weight-load ramp is a separate rising run before a
    plateau and does not join the allocation burst.
    """
    require(len(values) >= 2, "incomplete timeline for allocation burst")
    peak = max(range(1, len(values)), key=lambda i: values[i] - values[i - 1])
    if values[peak] <= values[peak - 1]:
        return 0
    start, end = peak - 1, peak
    while start > 0 and values[start - 1] < values[start]:
        start -= 1
    while end + 1 < len(values) and values[end + 1] > values[end]:
        end += 1
    return values[end] - values[start]


def envelope(review, observed, classified, guard_ceiling, review_digest, minimum_allowance=0):
    observed = byte_count(observed)
    classified = byte_count(classified)
    require(
        0 < observed <= classified,
        "physical observation exceeds classified measurement",
    )
    if review["basis"] == "measured_aggregate":
        require(
            set(review) == {"basis", "co_live_extra_bytes", "uncertainty_allowance_bytes"},
            "unexpected empirical price field",
        )
        byte_count(
            classified
            + byte_count(review["co_live_extra_bytes"])
            + byte_count(review["uncertainty_allowance_bytes"])
        )
        require(
            review["uncertainty_allowance_bytes"] >= minimum_allowance,
            "uncertainty allowance is below the measured allocation burst or decode budget",
        )
        return dict(
            review,
            observed_peak_bytes=observed,
            classified_peak_bytes=classified,
            review_sha256=review_digest,
        )
    require(
        review["basis"] == "guarded_ceiling"
        and set(review) == {"basis", "unsampled_overshoot_upper_bytes"},
        "unknown physical pricing basis",
    )
    ceiling = byte_count(guard_ceiling)
    upper = byte_count(ceiling + byte_count(review["unsampled_overshoot_upper_bytes"]))
    require(
        observed <= ceiling and upper >= classified,
        "guard cannot cover the verified measurement",
    )
    return dict(review, observed_peak_bytes=observed, guard_ceiling_bytes=ceiling)


def sample_values(samples, width, label):
    require(isinstance(samples, list) and len(samples) >= 2, f"incomplete {label} timeline")
    last = 0
    values = []
    for sample in samples:
        require(
            isinstance(sample, list)
            and len(sample) == width
            and type(sample[0]) is int
            and last < sample[0],
            f"invalid {label} timeline",
        )
        last = sample[0]
        value, *_ = (byte_count(cell) for cell in sample[1:])
        values.append(value)
    return values


def tenant_growth(guard, uuid, absolute, unknown, baseline, card, pool):
    """Read the child charge from the same supervisor sample as its whole-card upper bound."""
    samples = guard.get("samples")
    require(isinstance(samples, list), "missing supervisor CUDA samples")
    events = [
        event for event in samples if event.get("phase") == "cuda-global-used-upper-bound-sample"
    ]
    require(len(events) <= unknown, "unexpected supervisor CUDA sample count")

    def charge_values(charge, upper):
        charge, upper = byte_count(charge), byte_count(upper)
        require(
            max(0, upper - baseline) <= charge <= upper <= absolute,
            "inconsistent supervisor CUDA charge",
        )
        return charge, upper

    peak = absolute_peak = last = 0
    for event in events:
        charge, upper = charge_values(
            event.get("childGlobalUsedGrowthBytes"), event.get("globalUsedUpperBoundBytes")
        )
        require(
            event.get("cudaDeviceUuid") == uuid
            and type(event.get("at_ns")) is int
            and event["at_ns"] > last,
            "inconsistent supervisor CUDA charge or device",
        )
        byte_count(event.get("gpuFreeBytes"))
        peak = max(peak, charge)
        absolute_peak = max(absolute_peak, upper)
        last = event["at_ns"]

    observation = guard["fit_observation"]
    if "cuda_render_end" in observation:
        end = observation["cuda_render_end"]
        require(isinstance(end, dict), "missing CUDA render-end measurements")
        entries = end.get("pool_growth")
        require(
            isinstance(entries, list) and len(entries) == byte_count(end.get("pool_ticks")),
            "missing original charge at CUDA pool substitution",
        )
        previous = -1
        for entry in entries:
            index = byte_count(entry["card_index"])
            require(
                previous < index < len(card)
                and entry["cuda_uuid"] == uuid
                and (card[index - 1][0] if index else 0)
                < byte_count(entry["read_ns"])
                <= byte_count(entry["at_ns"])
                == card[index][0]
                and byte_count(entry["free_bytes"]) == card[index][1]
                and card[index][2] == pool,
                "crossed original CUDA charge or pool sample",
            )
            charge, upper = charge_values(
                entry["child_growth_bytes"], entry["global_used_upper_bytes"]
            )
            peak = max(peak, charge)
            absolute_peak = max(absolute_peak, upper)
            previous = index
        if "rejected_growth" in end:
            reserve = byte_count(guard["admission"]["gpuFreeReserveBytes"])
            cap = byte_count(guard["admission"]["childGpuCapBytes"])
            require(
                all(free >= reserve and used <= cap for _, free, used in card)
                and all(event["gpuFreeBytes"] >= reserve for event in events),
                "completed CUDA reading breached the capture guard",
            )
        rejected = end.get("rejected_growth", [])
        require(isinstance(rejected, list), "invalid rejected CUDA readings")
        for entry in rejected:
            require(entry["cuda_uuid"] == uuid, "rejected CUDA reading belongs to another device")
            charge, upper = charge_values(
                entry["child_growth_bytes"], entry["global_used_upper_bytes"]
            )
            peak = max(peak, charge)
            absolute_peak = max(absolute_peak, upper)
        require(absolute_peak == absolute, "original CUDA charges omit the card high-water mark")
        if entries:
            mark = end.get("pool")
            require(
                isinstance(mark, dict)
                and entries[0]["card_index"] == byte_count(mark.get("card_index"))
                and pool == byte_count(mark.get("pool_reserved_high_bytes")),
                "CUDA pool substitutions differ from the output marker",
            )
    return peak, len(events)


def validate_render_scope(scope, end, dedicated, misses, pool, card):
    """Bind the capture verifier's post-output exception to its unchanged raw Job/card records."""
    if any(misses):
        require(
            isinstance(scope, dict), "incomplete WDDM measurement without verified render scope"
        )
    else:
        require(scope is None, "unexpected WDDM render-scope exception")
    identity = {
        "pid": dedicated["pid"],
        "created_filetime": dedicated["created_filetime"],
        "cuda_uuid": dedicated["adapter"]["cuda_uuid"],
    }
    pool_mark, zero = end["pool"], end["zero"]
    require(
        isinstance(pool_mark, dict) and isinstance(zero, dict) and end["pool_ticks"] > 0,
        "missing WDDM render markers",
    )
    for mark, value in ((pool_mark, pool), (zero, 0)):
        index = byte_count(mark["card_index"])
        require(
            all(type(mark.get(k)) is type(v) and mark[k] == v for k, v in identity.items())
            and index < len(card)
            and byte_count(mark["at_ns"]) == card[index][0]
            and value == card[index][2],
            "crossed WDDM render marker or card sample",
        )
    require(
        pool_mark["basis"] == "verified-output-pool"
        and byte_count(pool_mark["pool_reserved_high_bytes"]) == pool
        and zero["basis"] == "owned-process-attributed"
        and pool_mark["card_index"] < zero["card_index"] == len(card) - 1
        and (zero["before"], zero["after"])
        in {
            ("live", "live"),
            ("live", "exited_empty"),
            ("exited_empty", "exited_empty"),
        },
        "invalid WDDM output or release marker",
    )
    if not any(misses):
        return
    expected = dict(
        identity,
        pool_card_index=pool_mark["card_index"],
        zero_card_index=zero["card_index"],
        dedicated_missing_after_output=misses[0],
        shared_missing_after_output=misses[1],
    )
    require(
        scope == expected and all(type(scope[k]) is type(v) for k, v in expected.items()),
        "verified render scope differs from original WDDM observations",
    )


def validate_rejected_growth(
    guard, end, native_pool, card, pool_high, *,
    completed=False, tied_read=False, fresh_gpu0=False, exit_at_zero=False,
):
    """A failed read is chargeable only when this Job's verified output pool has retired."""
    mark, zero = end["pool"], end["zero"]
    require(
        isinstance(native_pool, dict)
        and set(native_pool)
        == {
            "pid",
            "cudaDeviceUuid",
            "poolUsedBytes",
            "poolUsedHighBytes",
            "poolReservedBytes",
            "poolReservedHighBytes",
        },
        "missing original native CUDA pool",
    )
    used = byte_count(native_pool["poolUsedBytes"])
    reserved = byte_count(native_pool["poolReservedBytes"])
    require(
        byte_count(native_pool["pid"]) == mark["pid"]
        and native_pool["cudaDeviceUuid"] == mark["cuda_uuid"]
        and 0 < used <= reserved <= pool_high == byte_count(native_pool["poolReservedHighBytes"])
        and used <= byte_count(native_pool["poolUsedHighBytes"]) <= pool_high,
        "rejected observation differs from native CUDA pool or process",
    )
    previous = 0
    last_positive_read = 0
    exited_at = None
    reserve = byte_count(guard["admission"]["gpuFreeReserveBytes"])
    for event in end["rejected_growth"]:
        index = byte_count(event["next_card_index"])
        read, finished = byte_count(event["read_ns"]), byte_count(event["at_ns"])
        original = event["pool"]
        require(
            mark["card_index"] < index <= (len(card) if completed else zero["card_index"])
            and max(previous, card[index - 1][0]) < read <= finished
            and (index == len(card) or finished <= card[index][0])
            and byte_count(event["pid"]) == mark["pid"]
            and byte_count(event["created_filetime"]) == mark["created_filetime"]
            and event["before"] == "live"
            and event["after"] in ("live", "exited_empty")
            and exited_at is None
            and event["reason"] == "probe-failure"
            and original == native_pool
            and all(type(original[k]) is type(v) for k, v in native_pool.items())
            and byte_count(event["free_bytes"]) >= reserve
            and byte_count(event["child_growth_bytes"])
            <= byte_count(guard["admission"]["childGpuCapBytes"]),
            "unverified rejected CUDA reading or physical guard violation",
        )
        retired = (
            event["detail"] == "CUDA pool proof exceeds concurrent global use"
            and event["child_growth_bytes"] < reserved
        )
        exited = (
            event["detail"] == "CUDA pool proof has no valid owned PID/UUID allocation"
            and event["after"] == "exited_empty"
        )
        require(retired or exited, "unverified CUDA pool rejection reason")
        if event["after"] == "exited_empty":
            exited_at = index
        previous = finished
        if event["child_growth_bytes"]:
            last_positive_read = read

    if exited_at is not None:
        require(
            (zero is None or zero["before"] == "exited_empty")
            and not any(value for _, _, value in card[exited_at:]),
            "original CUDA Job resumed after exit",
        )

    if completed:
        # Job completion has a different terminal interval, validated by validate_completed_job.
        return
    # A rejected sample may extend live GPU usage beyond a previously plausible terminal PDH gap.
    last_used = max(last_positive_read, max(t for t, _, value in card if value > 0))
    if tied_read:
        events = end["rejected_growth"]
        final = events[-1]
        if exit_at_zero:
            dedicated = guard["cuda_attribution"]["wddm_dedicated"]["samples"]
            shared = guard["fit_observation"]["wddm_shared"]["samples"]
            # Each live poll has an ordered pair of PDH observations, except the final recorded gap.
            require(
                len(dedicated) == len(shared) == zero["card_index"] + len(events) - 1
                and all(d[0] <= s[0] for d, s in zip(dedicated, shared))
                and all(s[0] < d[0] for s, d in zip(shared, dedicated[1:])),
                "incomplete paired WDDM observations before the last live read",
            )
            terminal_states = (
                len(events) == 3
                and exited_at is None
                and final["after"] == "live"
                and (zero["before"], zero["after"]) == ("live", "exited_empty")
            )
        else:
            terminal_states = (
                exited_at == zero["card_index"]
                and final["after"] == "exited_empty"
                and zero["before"] == zero["after"] == "exited_empty"
            )
        require(
            len(events) >= 2
            and [v["card_index"] for v in end["pool_growth"]]
            == list(range(mark["card_index"], zero["card_index"]))
            and all(v["next_card_index"] == zero["card_index"] for v in events)
            and all(
                v["child_growth_bytes"] <= events[i]["child_growth_bytes"]
                for i, v in enumerate(events[1:])
            )
            and mark["at_ns"] < events[-2]["at_ns"] < final["read_ns"] == last_used
            and 0 < final["child_growth_bytes"] < reserved
            and final["before"] == "live"
            and terminal_states
            and final["detail"] == "CUDA pool proof exceeds concurrent global use"
            and final["next_card_index"] == zero["card_index"]
            and final["read_ns"] <= final["at_ns"] < zero["at_ns"],
            "terminal CUDA read does not precede this Job's measured zero",
        )
    validate_post_output_gaps(guard, end, last_used, tied_read=tied_read, fresh_gpu0=fresh_gpu0)


def validate_post_output_gaps(guard, end, last_used, *, tied_read=False, fresh_gpu0=False):
    observation = guard["fit_observation"]
    mark, zero = end["pool"], end["zero"]
    if tied_read:
        final = end["rejected_growth"][-1]
    gaps = []
    for name, series in (
        ("dedicated", guard["cuda_attribution"]["wddm_dedicated"]),
        ("shared", observation["wddm_shared"]),
    ):
        missed = [
            item for item in observation["wddm_retry"][name]["events"] if item["recovered"] is False
        ]
        require(
            len(missed) == series["missing_active_polls"],
            "unlocated WDDM gap during CUDA pool retry",
        )
        if tied_read:
            require(
                len(missed) == 1
                and len(observation["wddm_retry"][name]["events"])
                == observation["wddm_retry"][name]["attempts"]
                == 1,
                "terminal read lacks both original isolated gauge gaps",
            )
        for item in missed:
            start = byte_count(item["start_ns"])
            if tied_read or fresh_gpu0:
                tick_ns = byte_count(guard["admission"]["pollMillis"]) * 1_000_000
                finish = byte_count(item["end_ns"])
                lower, upper = (
                    byte_count(item["local_filetime_before"]),
                    byte_count(item["local_filetime_after"]),
                )
                require(
                    start > 0
                    and 0 < lower <= upper
                    and (upper - lower) * 100 <= tick_ns
                    and 0 <= finish - start <= tick_ns,
                    "invalid WDDM gap clocks",
                )
            if tied_read:
                gaps.append(item)
                require(
                    final["read_ns"] <= start <= finish <= final["at_ns"],
                    "terminal gauge retry falls outside its source attempt",
                )
            require(
                (start >= last_used if tied_read else start > max(mark["at_ns"], last_used))
                and start <= byte_count(item["end_ns"]) <= zero["at_ns"]
                and item["retried"] is True
                and item["retry_sample"] is None
                and item["previous"] == series["samples"][-1]
                and byte_count(item["local_filetime_before"]) >= series["samples"][-1][0],
                "WDDM gap overlaps a rejected CUDA reading",
            )
    if tied_read:
        dedicated, shared = gaps
        require(
            dedicated["start_ns"] == final["read_ns"]
            and dedicated["end_ns"] <= shared["start_ns"] <= shared["end_ns"] == final["at_ns"]
            and dedicated["local_filetime_after"] <= shared["local_filetime_before"],
            "synchronous terminal gauge order differs from the original CUDA read",
        )


def validate_job_release(guard, claim, basis, *, require_quiet=True):
    pid = guard["pid"]
    if basis == "owned-job-completion":
        expected = {
            "scope": "supervisor-owned-windows-job",
            "root_pid": pid,
            "root_returncode": 0,
            "waited_returncode": 0,
            "job_member_pids": [],
            "owned_pids": [],
        }
        for observed in (guard["owned_completion"], claim["owned_execution_completion"]):
            require(
                json.dumps(observed, sort_keys=True) == json.dumps(expected, sort_keys=True),
                "missing owned Job completion",
            )
    else:
        require(basis == "archived-supervisor-return", "unknown CUDA completion basis")
    # Native shared release permits other tenants after owned Job exit. The
    # exceptional render intervals additionally require five quiet samples.
    activity = claim["release_background_activity"]
    release = activity["samples"]
    verification = claim["release_metadata_verification"]
    baseline = claim["admission_baseline"]
    total = byte_count(baseline["admitted_total_mib"])
    physical = registered_card_index(claim["device_uuid"])
    require(
        baseline["uuid"] == claim["device_uuid"] == guard["admission"]["cudaDeviceUuid"]
        and byte_count(claim["physical_index"]) == physical
        and total > 0
        and activity["status"] == "retained-existing-samples"
        and activity["activity_profile"] == claim["activity_profile"]
        and activity["admission_headroom_is_diagnostic_only"] is True
        and 1 <= len(release) == byte_count(activity["sample_count"]) <= 5,
        "shared CUDA release differs from the selected device or sample window",
    )
    if verification == {"status": "verified"}:
        require(
            len(release) == 5
            and claim["release_utilization_percent"]
            == [s["selected_card"]["utilization_percent"] for s in release],
            "incomplete verified release series",
        )
    else:
        require(
            not require_quiet
            and verification.get("status") == "inconclusive-driver-query-timing"
            and isinstance(verification.get("error"), str)
            and "release_utilization_percent" not in claim
            and verification.get("diagnostic") == activity,
            "missing original shared GPU0 timing diagnostics",
        )
    # A quiet card holds no more than its own pre-admission idle memory and has zero
    # utilization. That level comes only from stable, idle admission samples of this card and
    # is capped at the driver-context bound (plus the display on GPU1). Without admission
    # samples a quiet card holds none.
    admitted = claim.get("admission_background_activity", {}).get("samples", [])
    idle = [byte_count(s["selected_card"]["used_mib"]) for s in admitted]
    require(
        all(
            byte_count(s["selected_card"]["index"]) == physical
            and byte_count(s["selected_card"]["utilization_percent"]) == 0
            for s in admitted
        )
        and (not idle or max(idle) - min(idle) <= IDLE_SPREAD_MIB),
        "admission samples are not a stable idle reading of this card",
    )
    idle_mib = min(
        min(idle, default=0),
        DRIVER_CONTEXT_IDLE_MIB + DISPLAY_IDLE_ALLOWANCE_MIB.get(claim["device_uuid"], 0),
    )
    for index, sample in enumerate(release, 1):
        card = sample["selected_card"]
        require(
            byte_count(sample["sample_number"]) == index
            and sample["predicates"]["unique_uuid"] is True
            and sample["selected_index_matches_admission"] is True
            and sample["post_owned_exit_primary_metadata_valid"] is True
            and byte_count(card["index"]) == physical
            and byte_count(card["total_mib"]) == total
            and byte_count(card["used_mib"]) <= total
            and byte_count(card["free_mib"]) <= total
            and card["used_mib"] + card["free_mib"] <= total + 2,
            "shared CUDA release sample differs from the claimed adapter capacity",
        )
        if require_quiet:
            require(
                card["used_mib"] <= idle_mib and byte_count(card["utilization_percent"]) == 0,
                "completed render scope lacks five quiet release samples",
            )


def validate_cuda_guard_samples(guard, end, card):
    cap, reserve = guard["admission"]["childGpuCapBytes"], guard["admission"]["gpuFreeReserveBytes"]
    values = [(free, used) for _, free, used in card]
    values.extend(
        (v["gpuFreeBytes"], v["childGlobalUsedGrowthBytes"])
        for v in guard["samples"]
        if v.get("phase") == "cuda-global-used-upper-bound-sample"
    )
    values.extend(
        (v["free_bytes"], v["child_growth_bytes"])
        for v in end["pool_growth"] + end["rejected_growth"]
    )
    require(
        all(free >= reserve and used <= cap for free, used in values),
        "completed CUDA Job exceeded guard",
    )


def validate_completed_job(guard, claim, summary, basis, pool, card, native_pool):
    """Verify observed load through output. The final Job release supplies no PID memory sample."""
    end = guard["fit_observation"]["cuda_render_end"]
    mark, events = end["pool"], end["rejected_growth"]
    pid, uuid = guard["pid"], guard["admission"]["cudaDeviceUuid"]
    require(
        end["zero"] is None
        and uuid in REGISTERED_CUDA_CARDS
        and mark["basis"] == "verified-output-pool"
        and mark["pid"] == pid
        and mark["cuda_uuid"] == uuid
        and byte_count(mark["card_index"]) < len(card)
        and (mark["at_ns"], mark["pool_reserved_high_bytes"]) == (card[mark["card_index"]][0], pool)
        and card[mark["card_index"]][2] == pool
        and claim["physical_index"] == registered_card_index(uuid)
        and claim["device_uuid"] == uuid
        and claim["lease_released"] is True
        and claim["post_execution_verified"] is True
        and claim["release_metadata_verification"] == {"status": "verified"},
        "CUDA completion does not identify the released output and selected device",
    )
    validate_job_release(guard, claim, basis)
    final = [v for v in events if v["next_card_index"] == len(card)]
    require(
        len(final) == 4
        and events[-4:] == final
        and [v["card_index"] for v in end["pool_growth"]]
        == list(range(mark["card_index"], len(card)))
        and all(
            v["child_growth_bytes"] <= final[i]["child_growth_bytes"]
            for i, v in enumerate(final[1:])
        ),
        "incomplete CUDA post-output attempt sequence",
    )
    validate_rejected_growth(guard, end, native_pool, card, pool, completed=True)
    validate_cuda_guard_samples(guard, end, card)
    expected = {
        "basis": basis,
        "pid": pid,
        "cuda_uuid": uuid,
        "created_filetime": mark["created_filetime"],
        "pool_card_index": mark["card_index"],
        "terminal_rejected_attempts": len(final),
    }
    observation = guard["fit_observation"]
    tick_ns = byte_count(guard["admission"]["pollMillis"]) * 1_000_000
    terminal_gaps = []
    for name, series in (
        ("dedicated", guard["cuda_attribution"]["wddm_dedicated"]),
        ("shared", observation["wddm_shared"]),
    ):
        gaps = [v for v in observation["wddm_retry"][name]["events"] if v["recovered"] is False]
        require(
            mark["created_filetime"] == series["created_filetime"]
            and len(gaps) == series["missing_active_polls"] <= 1
            and len(series["samples"]) + len(gaps) == len(card) + len(events),
            "unlocated measurements in completed CUDA Job",
        )
        for gap in gaps:
            terminal_gaps.append(gap)
            start, finish = gap["start_ns"], gap["end_ns"]
            lower, upper = gap["local_filetime_before"], gap["local_filetime_after"]
            require(
                all(type(v) is int and v > 0 for v in (start, finish, lower, upper))
                and 0 <= finish - start <= tick_ns
                and 0 <= (upper - lower) * 100 <= tick_ns
                and final[-2]["at_ns"]
                < final[-1]["read_ns"]
                <= gap["start_ns"]
                <= gap["end_ns"]
                <= final[-1]["at_ns"]
                and gap["retried"] is True
                and gap["retry_sample"] is None
                and gap["previous"] == series["samples"][-1]
                and gap["local_filetime_before"] >= series["samples"][-1][0],
                "WDDM gap overlaps measured render",
            )
        expected[name + "_missing_after_output"] = len(gaps)
    if len(terminal_gaps) == 2:
        dedicated, shared = terminal_gaps
        require(
            dedicated["end_ns"] <= shared["start_ns"]
            and dedicated["local_filetime_after"] <= shared["local_filetime_before"],
            "terminal WDDM observations contradict gauge order",
        )
    require(
        json.dumps(summary, sort_keys=True) == json.dumps(expected, sort_keys=True),
        "CUDA completion evidence differs from raw record",
    )


def windows_device_peak(
    guard,
    native,
    adapter,
    raw_gross,
    scope=None,
    native_pool=None,
    *,
    completion=None,
    tied_read=False,
    fresh_gpu0=False,
    exit_at_zero=False,
):
    admission = guard["admission"]
    baseline = byte_count(admission.get("gpuGlobalUsedBaselineBytes"))
    attribution = guard["cuda_attribution"]
    observation = guard["fit_observation"]
    dedicated = attribution["wddm_dedicated"]
    shared = observation["wddm_shared"]
    require(
        isinstance(adapter, dict)
        and isinstance(adapter.get("cuda_uuid"), str)
        and adapter["cuda_uuid"]
        and adapter["cuda_uuid"] == admission["cudaDeviceUuid"]
        and adapter.get("counter") == r"\GPU Process Memory(*)\Dedicated Usage"
        and observation["wddm_shared_counter"] == "GPU Process Memory/Shared Usage"
        and type(guard["pid"]) is int
        and guard["pid"] > 0
        and type(dedicated["created_filetime"]) is int
        and dedicated["created_filetime"] > 0,
        "invalid WDDM adapter or process identity",
    )
    misses = [byte_count(rows["missing_active_polls"]) for rows in (dedicated, shared)]
    for rows in (dedicated, shared):
        require(
            rows["adapter"] == adapter
            and type(rows["pid"]) is int
            and rows["pid"] == guard["pid"]
            and type(rows["created_filetime"]) is int
            and rows["created_filetime"] == dedicated["created_filetime"]
            and rows["error"] is None,
            "incomplete or crossed WDDM process measurements",
        )
    wddm = sample_values(dedicated["samples"], 2, "WDDM dedicated")
    require(sum(value > 0 for value in wddm) >= 2, "incomplete WDDM dedicated peak")
    # Host pricing retains the producer's separately classified Job/working-set plus WDDM
    # shared peak. Its samples must belong to the same verified process as the device series.
    sample_values(shared["samples"], 2, "WDDM shared")

    pid = byte_count(attribution["pid_peak_bytes"])
    pool = byte_count(native["allocator_peak"])
    fallback = byte_count(attribution["pool_fallback_peak_bytes"])
    absolute = byte_count(attribution["global_upper_bytes"])
    unknown = byte_count(attribution["unknown_pid_polls"])
    require(
        pool > 0
        and fallback <= pool
        and guard["peak_gpu_bytes"] == (max(pid, fallback) or None)
        and (
            (unknown > 0 and absolute > 0)
            or (unknown == 0 and pid > 0 and absolute == fallback == 0)
        ),
        "inconsistent owned-PID or whole-card CUDA measurement",
    )
    owned_or_floor = max(pool, pid, max(wddm), byte_count(native["axes"]["floor_device_bytes"]))
    card = observation["card_samples_monotonic_ns_free_owned_or_fallback_bytes"]
    free = sample_values(card, 3, "free-card")
    growth, growth_count = tenant_growth(
        guard, adapter["cuda_uuid"], absolute, unknown, baseline, card, pool
    )
    if "cuda_render_end" in observation:
        end = observation["cuda_render_end"]
        rejected = end.get("rejected_growth", [])
        require(
            byte_count(end.get("pool_ticks")) + growth_count + len(rejected) == unknown,
            "unaccounted supervisor CUDA fallback tick",
        )
        if completion is not None:
            require(
                type(guard["returncode"]) is int and guard["returncode"] == 0,
                "CUDA Job did not return successfully",
            )
            claim, summary, basis = completion
            require(
                scope is None and rejected and fallback == pool,
                "conflicting CUDA end-of-render evidence",
            )
            validate_completed_job(guard, claim, summary, basis, pool, card, native_pool)
        elif any(misses) or rejected:
            require(fallback == pool, "output marker differs from native pool fallback")
            if tied_read:
                require(
                    isinstance(scope, dict) and rejected,
                    "missing terminal CUDA scope or original read",
                )
                scope = scope.copy()
                final = rejected[-1]
                for field, raw in (
                    ("terminal_rejected_read_ns", "read_ns"),
                    ("terminal_rejected_at_ns", "at_ns"),
                ):
                    require(
                        byte_count(scope.pop(field, None)) == final[raw],
                        "certificate terminal read timestamp differs",
                    )
            validate_render_scope(scope, end, dedicated, misses, pool, card)
            if rejected:
                validate_rejected_growth(
                    guard, end, native_pool, card, pool, tied_read=tied_read, fresh_gpu0=fresh_gpu0,
                    exit_at_zero=exit_at_zero,
                )
            elif fresh_gpu0:
                validate_post_output_gaps(
                    guard, end, max(t for t, _, value in card if value > 0), fresh_gpu0=True
                )
        else:
            require(scope is None, "unexpected WDDM render-scope exception")
        classified = max(owned_or_floor, growth)
        require(
            classified == raw_gross and classified <= byte_count(admission["childGpuCapBytes"]),
            "device measurement differs from verified native quote or exceeds guard",
        )
        return classified
    # Older producers quote the absolute card peak and have no output marker. Keep their
    # strict gap-free requirement and their conservative checks against both card baselines.
    require(
        not any(misses) and scope is None and completion is None,
        "incomplete WDDM measurement without render evidence",
    )
    require(
        max(absolute, owned_or_floor) == raw_gross,
        "device measurement differs from verified native quote",
    )
    return max(owned_or_floor, growth, max(0, absolute - baseline), free[0] - min(free))


RECOVERY_CERTIFICATE_SHA256 = "dcc7c1eb186b001c75f4a9df2d49dd06501e8064be7d1818adb2ea59031c093f"
RECOVERED_SIX = {
    "run_id": "44935671433",
    "run_attempt": "1",
    "artifact_id": "14719969057",
    "inference_revision": "7f6c4cca3fc053030bd71f0bfce6886adb762eb6",
    "plan_sha256": "0bcc7e1047f960234f93ba23c04a2abfe99c2581287c4163224defe74e4751b7",
    "archive_sha256": "83985597d8d0cf2399b2311c1f1d737058cc1c1530b1a53fce6c79faee123969",
    "receipt_sha256": "750313a3cdb89dd4a0e4e49b31935949301b309fe9410f2bf0ce47cb4628575f",
}


FOUR_COUNT_CAPTURE = {
    "run_id": "44944284405",
    "run_attempt": "1",
    "artifact_id": "14725116132",
    "inference_revision": "f5c9ff80acd00334fd668e4ecc8d56be3347affd",
    "plan_sha256": "0f4b541c702bf45d607e138b56044fa8407a788b3b041d6e5f600b040097cf35",
    "archive_sha256": "39366924a8a5e09d4e1b0186615b6d3d5de7995ac18c87cad2f48ca918feb8e8",
    "receipt_sha256": "d423f72d816ac6145c7553623a26e99f67fd59db8e83ab228a89ede8698ce8af",
}
FOUR_COUNT_CERTIFICATES = {
    3: "c18632894d2d40c7d2b0eea519657714d873a611ca9e6f0845b0809dab4a024d",
    5: "bd8567c64d120f24e9821747d94b63faec3e5919930ce65dccb5a79f7df86fe7",
}


FIVE_COUNT_CAPTURE = {
    "run_id": "44969015053",
    "run_attempt": "1",
    "artifact_id": "14735122440",
    "inference_revision": "4fdd13cce7f590be0aed3c52a73a9cea9e6d1b54",
    "plan_sha256": "476e244074871e6e486107e4f65a515bf593a3c593d009c679a7ffbb2b2dee51",
    "archive_sha256": "ebb9d70b374f2e2e430fe292a6bf0f0d15d9c702142a657158d1bf525f17d388",
    "receipt_sha256": "d711f34b0c83b54b41b147921b9601fe999b74ccaec8823b6dceadcd6d9fc545",
}
FIVE_COUNT_CERTIFICATES = {3: "ca4696364a2a4c951b76f58879e0d25b7e48fdd4d6df5d9e7b605c313cbfd02d"}


def partial_count_row(receipt, review, archive, certificates, *, five_count=False):
    identity = FIVE_COUNT_CAPTURE if five_count else FOUR_COUNT_CAPTURE
    hashes = FIVE_COUNT_CERTIFICATES if five_count else FOUR_COUNT_CERTIFICATES
    count = review["references"]
    require(
        archive
        and sha256(archive.read_bytes()) == identity["archive_sha256"]
        and all(
            str(review[k]) == identity[k]
            for k in ("artifact_id", "receipt_sha256", "run_id", "run_attempt")
        )
        and receipt["contract"]["head"] == identity["inference_revision"]
        and sha256(json.dumps(receipt["contract"], sort_keys=True, separators=(",", ":")).encode())
        == identity["plan_sha256"]
        and receipt["accepted"] is False
        and count in (range(2, 7) if five_count else range(2, 6)),
        "different partial archive or review",
    )
    record = receipt["counts"][str(count)]
    if count not in hashes:
        require(
            record["accepted"] is True
            and certificates is None
            and "recovery_certificate_sha256" not in review,
            "original accepted count cannot use recovery",
        )
        return record["candidate_row_pending_physical_review"]
    require(
        certificates
        and record["accepted"] is False
        and record["telemetry_failure"] == "incomplete_wddm_samples"
        and "candidate_row_pending_physical_review" not in record,
        "missing original rejected count or separate certificate",
    )
    certificate_bytes = certificates.read_bytes()
    document = json.loads(certificate_bytes)
    cert = document if five_count else document[str(count)]
    digest = sha256((json.dumps(cert, sort_keys=True, separators=(",", ":")) + "\n").encode())
    require(
        cert["schema"] == ("scail2-quiet-gpu0-last-live-recovery-v1" if five_count
                           else "scail2-quiet-gpu0-owned-zero-recovery-v1")
        and cert["references"] == count
        and cert["capture"] == identity
        and digest == hashes[count] == review["recovery_certificate_sha256"]
        and (not five_count or sha256(certificate_bytes) == digest),
        "different owned-zero recovery certificate",
    )
    return cert["candidate_row_pending_physical_review"]


def recovered_row(receipt, review, certificate_path, archive_path):
    require(
        certificate_path and archive_path,
        "recovery requires both the original archive and native certificate",
    )
    certificate_bytes = certificate_path.read_bytes()
    certificate = json.loads(certificate_bytes)
    require(
        certificate.get("schema") == "scail2-quiet-gpu0-six-recovery-v1"
        and certificate["capture"] == RECOVERED_SIX
        and certificate["references"] == review["references"] == 6
        and all(
            str(review[k]) == RECOVERED_SIX[k]
            for k in ("artifact_id", "receipt_sha256", "run_id", "run_attempt")
        )
        and receipt["contract"]["head"] == RECOVERED_SIX["inference_revision"]
        and receipt["accepted"] is False
        and receipt["counts"]["6"]["accepted"] is False
        and "candidate_row_pending_physical_review" not in receipt["counts"]["6"]
        and sha256(archive_path.read_bytes()) == RECOVERED_SIX["archive_sha256"]
        and review.get("recovery_certificate_sha256")
        == sha256(certificate_bytes)
        == RECOVERY_CERTIFICATE_SHA256,
        "recovery certificate, original capture and independent review must agree",
    )
    return certificate["candidate_row_pending_physical_review"]


def validate_fresh_gpu0_plan(receipt, review, guard, record, *, five_count=False):
    """Match the hosted five-count owned-Job plan and the original verdict for every count.

    The plan may name GPU0, name another registered card, or name none (card-free selection)."""
    contract = receipt["contract"]
    request = receipt["request"]
    encode = lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    # The native scail2-plural-fit request, apart from its card, fixes the 84 GiB device cap,
    # 6 GiB free reserve, 64 GiB Job cap, 250 ms poll and original deadline. Naming GPU0 gives
    # 70d4d02ac74e5c0a9994b5728c4950df050c652bc47263fe3164ea6cbb601722.
    request_digest = "d28931338bf5350e2e408106562af9bc01a072f09f22ee0e857a1849ace0c90c"
    named = request["policy"].get("cudaDeviceUuid")
    card_free = dict(
        request, policy={k: v for k, v in request["policy"].items() if k != "cudaDeviceUuid"}
    )

    def original_verdict(count, item):
        if five_count and count == "3":
            return (
                item["accepted"] is False
                and item.get("telemetry_failure") == "incomplete_wddm_samples"
                and "candidate_row_pending_physical_review" not in item
            )
        return (
            item["accepted"] is True
            and "candidate_row_pending_physical_review" in item
            and "telemetry_failure" not in item
        )

    require(
        contract.get("cuda_completion") == "supervisor-owned-windows-job"
        and contract["backend"] == "candle-cuda"
        and contract["counts"] == [6, 2, 3, 4, 5]
        and "recovered_six" not in contract
        and request == contract["request"]
        and sha256(encode(card_free) + b"\n") == request_digest
        and (named is None) == ("cuda_card_selection" in contract)
        and (named is None or named in REGISTERED_CUDA_CARDS)
        and review.get("hosted_plan_sha256") == sha256(encode(contract))
        and receipt["accepted"] is (not five_count)
        and set(receipt["counts"]) == {str(count) for count in contract["counts"]}
        and all(original_verdict(count, item) for count, item in receipt["counts"].items()),
        "fresh capture differs from the complete accepted hosted owned-Job plan",
    )
    output = record["native"]["output"]
    body = record["body"]
    require(
        record["native"]["backend"] == "candle-cuda"
        and record["native"]["count"] == review["references"]
        and body["actual_devices"] == ["candle-cuda"]
        and body["body_completed"] == body["successful_libtest_summaries"] == 1
        and (output["width"], output["height"], output["frames"], output["fps"])
        == (832, 480, 45, 16)
        and re.fullmatch(r"[a-f0-9]{64}", output["sha256"]),
        "fresh GPU0 count lacks a completed native RGB output",
    )
    admission = guard["admission"]
    deadline = record["claim"]["claim_deadline_monotonic"]
    policy = executed_policy(receipt)
    require(
        type(deadline) in (int, float)
        and 0 < deadline == admission["originalClaimDeadlineMonotonic"]
        and all(
            admission[k] == policy[k]
            for k in (
                "backend",
                "cudaDeviceUuid",
                "childFootprintCapBytes",
                "hostFreeReserveBytes",
                "deadlineSeconds",
                "childGpuCapBytes",
                "gpuFreeReserveBytes",
                "pollMillis",
                "termGraceMillis",
            )
        ),
        "fresh capture differs from its original Job deadline or resource guard",
    )


def project(receipt_path, review_path, *, certificate_path=None, archive_path=None):
    receipt_bytes, review_bytes = receipt_path.read_bytes(), review_path.read_bytes()
    review = json.loads(review_bytes)
    require(
        sha256(receipt_bytes) == review["receipt_sha256"],
        "review belongs to a different receipt",
    )
    receipt = json.loads(receipt_bytes)
    five_count = (
        str(review["artifact_id"]) == FIVE_COUNT_CAPTURE["artifact_id"]
        or review["receipt_sha256"] == FIVE_COUNT_CAPTURE["receipt_sha256"]
    )
    partial = five_count or (
        str(review["artifact_id"]) == FOUR_COUNT_CAPTURE["artifact_id"]
        or review["receipt_sha256"] == FOUR_COUNT_CAPTURE["receipt_sha256"]
    )
    recovery = not partial and (certificate_path is not None or archive_path is not None)
    require(
        (receipt["accepted"] is True or recovery or partial)
        and receipt["contract"]["protocol"] == "scail2-plural-fit-v1",
        "incomplete or unrelated campaign",
    )
    require(
        str(review["run_id"]) == receipt["run_id"]
        and str(review["run_attempt"]) == receipt["run_attempt"],
        "review belongs to another workflow attempt",
    )
    if (
        str(review["artifact_id"]) == RECOVERED_SIX["artifact_id"]
        or review["receipt_sha256"] == RECOVERED_SIX["receipt_sha256"]
    ):
        require(recovery, "archived six can qualify only through the recovered count")
    count = review["references"]
    require(
        type(count) is int
        and 2 <= count <= 6
        and type(review["artifact_id"]) is int
        and review["artifact_id"] > 0,
        "missing artifact or character identity",
    )
    require(
        isinstance(review["rationale"], str) and review["rationale"].strip(),
        "missing physical review rationale",
    )
    record = receipt["counts"][str(count)]
    require(
        (record["accepted"] is True or recovery or partial)
        and record["claim"]["post_execution_verified"] is True,
        "capture did not complete",
    )
    guard_name = f"count-{count}.supervisor.json"
    guard_bytes = (receipt_path.parent / guard_name).read_bytes()
    require(
        sha256(guard_bytes) == receipt["logs"][guard_name]["sha256"],
        "physical guard record changed",
    )
    guard = json.loads(guard_bytes)
    require(
        guard["returncode"] == 0 and guard["pid"] == record["native"]["pid"],
        "capture guard did not complete for this process",
    )
    if partial:
        row = partial_count_row(receipt, review, archive_path, certificate_path, five_count=five_count)
    elif recovery:
        row = recovered_row(receipt, review, certificate_path, archive_path)
    else:
        row = record["candidate_row_pending_physical_review"]
        require(
            "recovery_certificate_sha256" not in review,
            "unexpected recovery review for ordinary capture",
        )
    native, witness = record["native"], row["witness"]
    require(
        witness["axes"] == native["axes"] and native["axes"]["references"] == count,
        "crossed native axes",
    )
    tokens = native["prompt_tokens"]
    require(
        tokens == [512] * (2 if native["axes"]["classifier_free_guidance"] else 1)
        and witness["text_tokens"] == min(tokens),
        "incomplete text coverage",
    )
    require(
        row["inference_revision"] == receipt["contract"]["head"]
        and re.fullmatch(r"[0-9a-f]{40}", row["inference_revision"]),
        "crossed inference revision",
    )
    admission, hw = guard["admission"], receipt["hardware"]
    fresh_gpu0 = "cuda_completion" in receipt["contract"] or "hosted_plan_sha256" in review
    if fresh_gpu0:
        require(
            not recovery
            and (not partial or five_count)
            and row["platform"] == "windows"
            and str(hw["index"]) == str(registered_card_index(hw["uuid"])),
            "expected an ordinary Windows capture on a registered card",
        )
        validate_fresh_gpu0_plan(receipt, review, guard, record, five_count=five_count)
    if row["platform"] == "macos":
        require(
            (row["device"], row["runtime"]) == (hw["chip"], hw["os_build"])
            and native["axes"]["backend"] == "mlx",
            "crossed Metal identity",
        )
        pools = {
            "unified": (
                guard["lifetime_peak_host_bytes"],
                witness["gross_unified_bytes"],
                admission["childFootprintCapBytes"],
            )
        }
        bounds = {
            "measurement": "darwin_lifetime_footprint",
            "mlx_decode_budget_bytes": witness["mlx_decode_budget_bytes"],
        }
        require(
            byte_count(bounds["mlx_decode_budget_bytes"]) > 0,
            "missing decoder measurement cap",
        )
    else:
        require(
            row["platform"] == "windows"
            and (row["device"], row["runtime"]) == (hw["name"], hw["driver_version"])
            and native["axes"]["backend"] == "cuda"
            and hw["uuid"] == receipt["adapter"]["cuda_uuid"]
            and re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", hw["uuid"])
            and re.fullmatch(r"luid_0x[0-9a-f]{8}_0x[0-9a-f]{8}", receipt["adapter"]["luid"])
            and receipt["adapter"]["luid"] != "luid_0x00000000_0x00000000"
            and type(receipt["adapter"]["physical_node"]) is int
            and 0 <= receipt["adapter"]["physical_node"] < 32
            and type(receipt["adapter"]["node_mask"]) is int
            and receipt["adapter"]["node_mask"] == 1 << receipt["adapter"]["physical_node"],
            "crossed CUDA identity",
        )
        validate_card_selection(receipt, record)
        # GPU0 captures, and any owned-Job or card-selected capture on another registered card,
        # bind the native ordinal-zero adapter to the recorded card. Only GPU0's PCI address is
        # pinned; another card's is bound by NVML, WDDM and the native ordinal agreeing.
        if (
            str(hw.get("index")) == "0"
            or hw["uuid"] == GPU0
            or fresh_gpu0
            or "cuda_card_selection" in receipt["contract"]
        ):
            adapter = receipt["adapter"]
            pci = str(hw.get("pci.bus_id", "")).lower()
            require(
                str(hw["index"]) == str(registered_card_index(hw["uuid"]))
                and isinstance(native.get("cuda_device"), dict)
                and all(
                    type(native["cuda_device"].get(k)) is int
                    for k in ("ordinal", "visible_count", "node_mask")
                )
                and re.fullmatch(r"[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci)
                and (hw["uuid"] != GPU0 or pci == GPU0_PCI)
                and native.get("cuda_device")
                == {
                    "ordinal": 0,
                    "visible_count": 1,
                    "uuid": hw["uuid"],
                    "pci_bus_id": pci,
                    "luid": adapter["luid"],
                    "node_mask": adapter["node_mask"],
                },
                "CUDA capture lacks the matching native execution adapter",
            )
        native_pool = None
        end = guard["fit_observation"].get("cuda_render_end", {})
        if fresh_gpu0 or "rejected_growth" in end:
            policy = executed_policy(receipt)
            policy_bytes = (
                json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode()
            require(
                receipt["request"]["policy"] == receipt["contract"]["request"]["policy"]
                and admission["policySha256"] == sha256(policy_bytes)
                and all(
                    admission[k] == policy[k] and type(admission[k]) is type(policy[k])
                    for k in ("cudaDeviceUuid", "childGpuCapBytes", "gpuFreeReserveBytes")
                ),
                "CUDA rejection guard differs from approved capture policy",
            )
        if end.get("rejected_growth"):
            pool_name = f"count-{count}.cuda-pool.json"
            pool_bytes = (receipt_path.parent / pool_name).read_bytes()
            require(
                sha256(pool_bytes) == receipt["logs"][pool_name]["sha256"],
                "native CUDA pool record changed",
            )
            native_pool = json.loads(pool_bytes)
        scope = row.get("cuda_render_scope")
        tied_read = (partial and not record["accepted"]) or (
            fresh_gpu0
            and scope is not None
            and ("terminal_rejected_read_ns" in scope or "terminal_rejected_at_ns" in scope)
        )
        completion = None
        if recovery or "cuda_completed_job" in row:
            summary = row["cuda_completed_job"]
            basis = "archived-supervisor-return" if recovery else "owned-job-completion"
            require(
                summary["basis"] == basis
                and str(hw["index"]) == str(registered_card_index(hw["uuid"])),
                "unverified CUDA completion source",
            )
            require(
                review["domains"]["device"]["basis"] == "measured_aggregate",
                "the completed RGB device interval requires an empirical review",
            )
            require(
                type(guard["admission"]["pollMillis"]) is int
                and guard["admission"]["pollMillis"]
                == receipt["request"]["policy"]["pollMillis"]
                == 250,
                "completed Job polling interval differs from the signed policy",
            )
            completion = (record["claim"], summary, basis)
        if partial or fresh_gpu0 or completion:
            require(
                byte_count(record["claim"]["admission_baseline"]["admitted_total_mib"])
                == int(hw["memory.total"]),
                "release baseline differs from the physical card capacity",
            )
        if (partial or fresh_gpu0) and not completion:
            require(
                guard["admission"]["pollMillis"]
                == receipt["request"]["policy"]["pollMillis"]
                == 250
                and record["claim"]["device_uuid"] == hw["uuid"]
                and record["claim"]["physical_index"] == registered_card_index(hw["uuid"])
                and record["claim"]["lease_released"] is True
                and review["domains"]["device"]["basis"] == "measured_aggregate",
                "CUDA count lacks its original signed guard or release",
            )
            validate_job_release(
                guard,
                record["claim"],
                "owned-job-completion",
                require_quiet=tied_read,
            )
            validate_cuda_guard_samples(
                guard,
                end,
                guard["fit_observation"]["card_samples_monotonic_ns_free_owned_or_fallback_bytes"],
            )
        if completion or partial or fresh_gpu0:
            counters = guard["fit_observation"]["windows_peak"]
            host = max(
                byte_count(guard["peak_host_bytes"]),
                *(
                    byte_count(counters[k])
                    for k in (
                        "peak_job_commit_bytes",
                        "peak_private_commit_bytes",
                        "peak_working_set_bytes",
                    )
                ),
            )
            shared = sample_values(
                guard["fit_observation"]["wddm_shared"]["samples"], 2, "WDDM shared"
            )
            require(
                witness["gross_host_bytes"]
                == max(host + max(shared), native["axes"]["floor_host_bytes"])
                and (
                    not fresh_gpu0
                    or witness["gross_host_bytes"]
                    <= byte_count(admission["childFootprintCapBytes"])
                )
                and witness["gross_unified_bytes"] == 0
                and witness["mlx_decode_budget_bytes"] is None,
                "completed CUDA host bounds differ from original measurements",
            )
        pools = {
            "host": (
                guard["peak_host_bytes"],
                witness["gross_host_bytes"],
                admission["childFootprintCapBytes"],
            ),
            "device": (
                native["allocator_peak"],
                windows_device_peak(
                    guard,
                    native,
                    receipt["adapter"],
                    witness["gross_device_bytes"],
                    scope,
                    native_pool,
                    completion=completion,
                    tied_read=tied_read,
                    fresh_gpu0=fresh_gpu0,
                    exit_at_zero=five_count and count == 3,
                ),
                admission["childGpuCapBytes"],
            ),
        }
        bounds = {
            "measurement": "windows_job_and_wddm",
            **(
                {"device_interval": "load_through_completed_rgb"}
                if completion or tied_read or (fresh_gpu0 and scope is not None)
                else {}
            ),
            # Provenance: the physical card that ran this capture.
            "capture_adapter": {
                k: receipt["adapter"][k] for k in ("cuda_uuid", "luid", "node_mask")
            },
            # Hardware class with device and runtime: workers on any identical card match it.
            "device_total_mib": card_total_mib(hw),
        }
    require(set(review["domains"]) == set(pools), "review is missing a physical domain")
    # A measured aggregate allows for growth between samples: the whole allocation burst in
    # the owned WDDM dedicated or card series (device) and the WDDM shared series (host).
    # An MLX price never falls below the native decode budget.
    if row["platform"] == "macos":
        minimums = {"unified": max(0, bounds["mlx_decode_budget_bytes"] - pools["unified"][1])}
    else:
        observation = guard["fit_observation"]
        minimums = {
            "device": max(
                burst_growth(
                    sample_values(guard["cuda_attribution"]["wddm_dedicated"]["samples"], 2, "WDDM dedicated")
                ),
                burst_growth(
                    [
                        byte_count(sample[2])
                        for sample in observation["card_samples_monotonic_ns_free_owned_or_fallback_bytes"]
                    ]
                ),
            ),
            "host": burst_growth(sample_values(observation["wddm_shared"]["samples"], 2, "WDDM shared")),
        }
    for domain, (observed, classified, ceiling) in pools.items():
        bounds[domain] = envelope(
            review["domains"][domain],
            observed,
            classified,
            ceiling,
            sha256(review_bytes),
            minimums[domain],
        )
    require(
        re.fullmatch(r"[0-9a-f]{64}", admission["policySha256"]),
        "missing guard policy digest",
    )
    return {
        **{k: row[k] for k in ("platform", "device", "runtime", "inference_revision")},
        "artifact_id": review["artifact_id"],
        "receipt_sha256": review["receipt_sha256"],
        "guard_policy_sha256": admission["policySha256"],
        "accepted": receipt["accepted"],
        **({"count_accepted": record["accepted"]} if partial else {}),
        **(
            {"recovery_certificate_sha256": review["recovery_certificate_sha256"]}
            if recovery or (partial and not record["accepted"])
            else {}
        ),
        "post_execution_verified": record["claim"]["post_execution_verified"],
        "guard_exit_code": guard["returncode"],
        "axes": native["axes"],
        "prompt_tokens": tokens,
        "bounds": bounds,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("--recovery-certificate", type=Path)
    parser.add_argument("--recovery-archive", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            project(
                args.receipt,
                args.review,
                certificate_path=args.recovery_certificate,
                archive_path=args.recovery_archive,
            ),
            sort_keys=True,
            indent=2,
        )
    )
