# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.draft_window_recovery import plan_draft_window_recovery

pytestmark = pytest.mark.skip_global_cleanup


def test_recovery_replays_from_first_hole_but_transfers_exact_ranges():
    plan = plan_draft_window_recovery(
        request_id="request-0",
        main_cached_end=1024,
        joint_cached_end=512,
        consumer_boundary=1024,
        window_sizes=(512, 256),
        block_sizes=(64, 64),
        query_len=1,
        max_model_len=2048,
        page_rounded=(False, False),
        draft_ready_ranges=(
            ((512, 640), (704, 1024)),
            ((768, 896), (960, 1024)),
        ),
    )

    assert plan.required_ranges == ((512, 1024), (768, 1024))
    assert plan.replay_ranges == ((640, 1024),)
    assert plan.missing_ranges_by_group == (
        ((640, 704),),
        ((896, 960),),
    )
    assert plan.saved_vs_common_tokens == 128
