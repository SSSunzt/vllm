# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from tests.v1.core.utils import (
    create_requests,
    create_scheduler,
    new_kv_cache_spec,
    new_sliding_window_spec,
)
from vllm.v1.core.draft_window_recovery import plan_draft_window_recovery
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
)
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu.model_states.default import (
    _protect_draft_recovery_main_slots_cpu,
)


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


def test_recovery_protects_main_slots_but_keeps_draft_writes():
    slot_mappings = torch.arange(16, dtype=torch.int64)
    slot_mappings = slot_mappings.reshape(2, 8)
    positions = torch.arange(56, 64)

    _protect_draft_recovery_main_slots_cpu(
        slot_mappings=slot_mappings,
        query_start_loc=(0, 8),
        positions=positions,
        main_cached_ends=(60,),
        main_group_ids=(0,),
        pad_slot_id=-1,
    )

    assert slot_mappings[0].tolist() == [-1, -1, -1, -1, 4, 5, 6, 7]
    assert slot_mappings[1].tolist() == list(range(8, 16))


def test_scheduler_rewinds_only_draft_window_hole():
    block_size = 4
    base = create_scheduler(block_size=block_size, enable_prefix_caching=True)
    kv_cache_config = KVCacheConfig(
        num_blocks=128,
        num_blocks_by_pool=[128, 32],
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["target"],
                new_kv_cache_spec(block_size=block_size),
                block_pool_id=0,
            ),
            KVCacheGroupSpec(
                ["draft"],
                new_sliding_window_spec(block_size=block_size, sliding_window=16),
                block_pool_id=1,
                role=KVCacheGroupRole.DRAFT_WINDOW,
            ),
        ],
    )
    scheduler = Scheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )
    manager = scheduler.kv_cache_manager
    producer, consumer = create_requests(
        num_requests=2,
        num_tokens=64,
        same_prompt=True,
        block_size=block_size,
    )
    assert manager.allocate_slots(producer, producer.num_tokens) is not None
    producer.num_computed_tokens = producer.num_tokens
    manager.remove_skipped_blocks(
        producer.request_id,
        producer.num_computed_tokens,
        num_prompt_tokens=producer.num_prompt_tokens,
    )
    manager.cache_blocks(producer, producer.num_computed_tokens)
    missing_block_id = manager.get_blocks(producer.request_id).blocks[1][14].block_id
    manager.free(producer)
    manager.block_pools[1].evict_blocks({missing_block_id})

    scheduler.add_request(consumer)
    output = scheduler.schedule()
    new_request = next(
        request
        for request in output.scheduled_new_reqs
        if request.req_id == consumer.request_id
    )

    assert new_request.num_computed_tokens == 56
    assert new_request.replay_start == 56
    assert new_request.draft_recovery_main_cached_end == 60
    assert output.num_scheduled_tokens[consumer.request_id] == 8


def test_scheduler_clears_stale_draft_recovery_protection():
    scheduler = create_scheduler(block_size=4, enable_prefix_caching=True)
    request = create_requests(num_requests=1, num_tokens=8, block_size=4)[0]
    request.draft_recovery_main_cached_end = 4

    assert scheduler._mark_draft_window_recovery(request, 4) == 0
    assert request.draft_recovery_main_cached_end == 0


def test_async_load_reservations_remain_scoped_to_each_pool():
    request = create_requests(num_requests=1, num_tokens=8, block_size=4)[0]
    coordinator = SimpleNamespace(
        get_num_blocks_to_allocate_by_pool=lambda **kwargs: (8, 2)
    )
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.max_model_len = 16
    scheduler.num_spec_tokens = 0
    scheduler.num_lookahead_tokens = 0
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=coordinator,
        block_pools=(object(), object()),
        empty_kv_cache_blocks=SimpleNamespace(blocks=()),
    )
    scheduler._inflight_prefills = {request}

    assert scheduler._request_remaining_blocks_by_pool(request) == (8, 2)
    assert scheduler._inflight_prefill_reserved_blocks() == (8, 2)
