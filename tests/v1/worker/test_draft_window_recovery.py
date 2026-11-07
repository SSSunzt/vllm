# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.model_states.default import (
    _protect_draft_recovery_main_slots_kernel,
    protect_draft_recovery_main_slots,
)


def test_recovery_masks_target_slots_but_keeps_drafter_writes():
    slot_mappings = torch.arange(16, dtype=torch.int64).reshape(2, 8)

    protect_draft_recovery_main_slots(
        slot_mappings=slot_mappings,
        query_start_loc=(0, 8),
        positions=torch.arange(56, 64),
        main_cached_ends=(60,),
        main_group_ids=(0,),
    )

    assert slot_mappings[0].tolist() == [-1, -1, -1, -1, 4, 5, 6, 7]
    assert slot_mappings[1].tolist() == list(range(8, 16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_recovery_slot_protection_kernel_on_gpu():
    slot_mappings = torch.arange(
        16, dtype=torch.int64, device="cuda"
    ).reshape(2, 8)
    query_start_loc = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    positions = torch.arange(56, 64, dtype=torch.int64, device="cuda")
    main_cached_ends = torch.tensor([60], dtype=torch.int32, device="cuda")
    main_group_ids = torch.tensor([0], dtype=torch.int32, device="cuda")

    _protect_draft_recovery_main_slots_kernel[(1, 1)](
        slot_mappings,
        slot_mappings.stride(0),
        main_group_ids,
        query_start_loc,
        positions,
        main_cached_ends,
        -1,
        NUM_MAIN_GROUPS=1,
        BLOCK=1024,
    )
    torch.cuda.synchronize()

    assert slot_mappings[0].tolist() == [-1, -1, -1, -1, 4, 5, 6, 7]
    assert slot_mappings[1].tolist() == list(range(8, 16))
