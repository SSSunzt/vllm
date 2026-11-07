# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recovery planning for independently stored draft attention windows."""

from dataclasses import dataclass

TokenRange = tuple[int, int]


def _replay_suffix(
    required: TokenRange, ready_ranges: tuple[TokenRange, ...]
) -> TokenRange | None:
    """Keep a ready prefix, then replay continuously from its first hole."""
    cursor, end = required
    for ready_start, ready_end in sorted(ready_ranges):
        if ready_end <= cursor:
            continue
        if ready_start > cursor or cursor >= end:
            break
        cursor = max(cursor, min(ready_end, end))
    return (cursor, end) if cursor < end else None


def _missing_ranges(
    required: TokenRange, ready_ranges: tuple[TokenRange, ...]
) -> tuple[TokenRange, ...]:
    """Subtract ready ranges without joining the remaining holes."""
    cursor, end = required
    missing: list[TokenRange] = []
    for ready_start, ready_end in sorted(ready_ranges):
        if ready_end <= cursor:
            continue
        if ready_start >= end:
            break
        if ready_start > cursor:
            missing.append((cursor, min(ready_start, end)))
        cursor = max(cursor, min(ready_end, end))
        if cursor >= end:
            break
    if cursor < end:
        missing.append((cursor, end))
    return tuple(missing)


@dataclass(frozen=True)
class DraftWindowRecoveryPlan:
    """Target replay and sparse transfer ranges needed by draft groups."""

    request_id: str
    main_cached_end: int
    joint_cached_end: int
    consumer_boundary: int
    required_ranges: tuple[TokenRange, ...]
    draft_ready_ranges: tuple[tuple[TokenRange, ...], ...]
    replay_ranges: tuple[TokenRange, ...]

    @property
    def common_endpoint_replay_tokens(self) -> int:
        return max(0, self.main_cached_end - self.joint_cached_end)

    @property
    def window_replay_tokens(self) -> int:
        return sum(end - start for start, end in self.replay_ranges)

    @property
    def missing_ranges_by_group(self) -> tuple[tuple[TokenRange, ...], ...]:
        historical_end = min(self.main_cached_end, self.consumer_boundary)
        return tuple(
            _missing_ranges(
                (required_start, min(required_end, historical_end)), ready_ranges
            )
            for (required_start, required_end), ready_ranges in zip(
                self.required_ranges, self.draft_ready_ranges, strict=True
            )
        )

    @property
    def saved_vs_common_tokens(self) -> int:
        return max(0, self.common_endpoint_replay_tokens - self.window_replay_tokens)


def plan_draft_window_recovery(
    *,
    request_id: str,
    main_cached_end: int,
    joint_cached_end: int,
    consumer_boundary: int,
    window_sizes: tuple[int, ...],
    block_sizes: tuple[int, ...],
    query_len: int,
    max_model_len: int,
    page_rounded: tuple[bool, ...],
    draft_ready_ranges: tuple[tuple[TokenRange, ...], ...],
) -> DraftWindowRecoveryPlan:
    """Plan the bounded historical context needed at a draft boundary."""
    num_groups = len(window_sizes)
    if not (
        num_groups == len(block_sizes) == len(page_rounded) == len(draft_ready_ranges)
    ):
        raise ValueError("Draft recovery inputs must have one entry per group")
    if query_len <= 0 or max_model_len <= 0:
        raise ValueError("query_len and max_model_len must be positive")
    if any(window <= 0 for window in window_sizes) or any(
        block <= 0 for block in block_sizes
    ):
        raise ValueError("Draft windows and block sizes must be positive")

    required_ranges: list[TokenRange] = []
    group_replays: list[TokenRange] = []
    historical_end = min(main_cached_end, consumer_boundary)
    for window, block, rounded, ready in zip(
        window_sizes,
        block_sizes,
        page_rounded,
        draft_ready_ranges,
        strict=True,
    ):
        if rounded:
            seq_len = min(consumer_boundary + query_len, max_model_len)
            required_start = seq_len - window - query_len + 1
        else:
            required_start = consumer_boundary - window + 1
        required_start = max(0, required_start) // block * block
        required = (required_start, consumer_boundary)
        required_ranges.append(required)
        replay = _replay_suffix((required_start, historical_end), ready)
        if replay is not None:
            group_replays.append(replay)

    replay_ranges = (
        ((min(start for start, _ in group_replays), historical_end),)
        if group_replays
        else ()
    )
    return DraftWindowRecoveryPlan(
        request_id=request_id,
        main_cached_end=main_cached_end,
        joint_cached_end=joint_cached_end,
        consumer_boundary=consumer_boundary,
        required_ranges=tuple(required_ranges),
        draft_ready_ranges=draft_ready_ranges,
        replay_ranges=replay_ranges,
    )
