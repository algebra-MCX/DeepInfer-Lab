"""CPU-only scheduler tests; the GPU is simulated by feeding back fake sampled tokens."""

import pytest

from minideepinfer.engine.block_manager import BlockManager
from minideepinfer.engine.scheduler import Scheduler
from minideepinfer.sampling_params import SamplingParams
from minideepinfer.sequence import Sequence, SequenceStatus

BLOCK = 4
EOS = 99


def make_scheduler(num_blocks=64, **kwargs):
    args = dict(max_num_seqs=8, max_num_batched_tokens=32, max_model_len=64, prefill_attempt_interval=1, eos_token_id=EOS)
    args.update(kwargs)
    return Scheduler(BlockManager(num_blocks), BlockManager(args["max_num_seqs"] + 2), **args)


def make_seq(prompt_len, max_tokens=4, ignore_eos=True):
    sp = SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos, seq_seed=0)
    return Sequence(list(range(prompt_len)), sp, BLOCK)


def run(sched, zos, token_fn=lambda seq: 7, max_steps=1000):
    """Drive the scheduler like LLMEngine.step() and return all SchedulerOutputs.

    `token_fn` is evaluated at schedule time, i.e. after the new token has been reserved.
    """
    outputs, inflight = [], None
    for _ in range(max_steps):
        if not sched.has_unfinished_seqs() and inflight is None:
            return outputs
        out = sched.schedule()
        outputs.append(out)
        handle = None if out.is_empty else (out.sample_seqs, [token_fn(seq) for seq, _ in out.sample_seqs])
        if inflight is not None:
            sched.process_outputs(*inflight)
            inflight = None
        if handle is not None:
            if zos:
                inflight = handle
            else:
                sched.process_outputs(*handle)
    raise AssertionError("did not finish")


def test_prefill_then_decode():
    sched = make_scheduler()
    seqs = [make_seq(5), make_seq(7)]
    for s in seqs:
        sched.add_sequence(s)
    out = sched.schedule()
    assert out.is_prompt and out.seqlens_q == [5, 7]
    assert all(s.num_inflight_tokens == 1 for s in seqs)
    sched.process_outputs(out.sample_seqs, [1, 2])
    out = sched.schedule()
    assert not out.is_prompt and out.seqlens_q == [1, 1]
    assert [s.num_cached_tokens for s in seqs] == [6, 8]


def test_chunked_prefill():
    sched = make_scheduler(max_num_batched_tokens=8)
    seq = make_seq(20)
    sched.add_sequence(seq)
    lens = []
    for _ in range(3):
        out = sched.schedule()
        assert out.is_prompt
        lens.append(out.seqlens_q[0])
        assert bool(out.sample_seqs) == (len(lens) == 3)
    assert lens == [8, 8, 4]
    assert seq.status == SequenceStatus.RUNNING and seq.num_cached_tokens == 20


def test_ignored_prompt():
    sched = make_scheduler(max_model_len=16)
    seq = make_seq(17)
    sched.add_sequence(seq)
    out = sched.schedule()
    assert out.ignored_seqs == [seq] and seq.status == SequenceStatus.FINISHED_IGNORED


def test_zos_lengths_and_no_wasted_steps_at_max_tokens():
    for zos in (False, True):
        sched = make_scheduler()
        seqs = [make_seq(5, max_tokens=6), make_seq(9, max_tokens=3)]
        for s in seqs:
            sched.add_sequence(s)
        outputs = run(sched, zos)
        assert [len(s.output_token_ids) for s in seqs] == [6, 3]
        assert all(s.status == SequenceStatus.FINISHED_LENGTH for s in seqs)
        # Exactly max_tokens samples per sequence: no step is wasted because of ZOS.
        for s in seqs:
            assert sum(1 for o in outputs for x, _ in o.sample_seqs if x is s) == s.sampling_params.max_tokens
        assert sched.block_manager.num_free_blocks == sched.block_manager.num_usable_blocks


def test_zos_eos_discards_extra_token():
    sched = make_scheduler()
    seq = make_seq(5, max_tokens=10, ignore_eos=False)
    sched.add_sequence(seq)
    run(sched, zos=True, token_fn=lambda s: EOS if s.get_output_len() >= 3 else 7)
    assert seq.status == SequenceStatus.FINISHED_STOPPED
    assert seq.output_token_ids == [7, 7, EOS]


def test_custom_stop_token_finishes_request():
    sched = make_scheduler()
    seq = Sequence(
        list(range(5)),
        SamplingParams(max_tokens=10, ignore_eos=True, stop_token_ids=(42,), seq_seed=0),
        BLOCK,
        request_id="stop-me",
    )
    sched.add_sequence(seq)
    out = sched.schedule()
    sched.process_outputs(out.sample_seqs, [42])
    assert seq.status == SequenceStatus.FINISHED_STOPPED
    assert seq.output_token_ids == [42]
    assert sched.abort_request("stop-me") is None


def test_abort_waiting_and_running_releases_both_resources():
    sched = make_scheduler()
    waiting = make_seq(5)
    running = Sequence(list(range(5)), SamplingParams(max_tokens=4, seq_seed=0), BLOCK, request_id="running")
    sched.add_sequence(waiting)
    sched.add_sequence(running)
    # Admit both, then put a third request behind them to exercise the waiting path.
    out = sched.schedule()
    sched.process_outputs(out.sample_seqs, [7, 7])
    queued = Sequence(list(range(4)), SamplingParams(max_tokens=4, seq_seed=0), BLOCK, request_id="queued")
    sched.add_sequence(queued)
    before = sched.snapshot()
    assert sched.abort_request("queued") is queued
    assert queued.status == SequenceStatus.FINISHED_ABORTED
    assert sched.snapshot()["waiting"] == before["waiting"] - 1

    blocks_before = sched.block_manager.num_free_blocks
    slots_before = sched.state_manager.num_free_blocks
    assert sched.abort_request("running") is running
    assert running.status == SequenceStatus.FINISHED_ABORTED
    assert sched.block_manager.num_free_blocks > blocks_before
    assert sched.state_manager.num_free_blocks > slots_before
    assert sched.abort_request("missing") is None


def test_preemption_recompute():
    # 1 + 6 usable blocks; two seqs of 8 prompt tokens growing to 16 tokens need 8 blocks in total.
    sched = make_scheduler(num_blocks=7)
    seqs = [make_seq(8, max_tokens=8), make_seq(8, max_tokens=8)]
    for s in seqs:
        sched.add_sequence(s)
    # The sampled token is a function of its position, as with deterministic sampling.
    run(sched, zos=True, token_fn=lambda s: 1000 + s.num_cached_tokens)
    assert sched.num_preempted > 0
    for s in seqs:
        assert s.output_token_ids == [1000 + p for p in range(8, 16)]
    assert sched.block_manager.num_free_blocks == 6


def test_preemption_keeps_chunked_prefill_at_head():
    # 9 usable blocks: a decoding seq (2 blocks) is preempted while a chunked prefill holds 7.
    # Requeueing it ahead of the chunked prefill would wait forever for blocks that request holds.
    sched = make_scheduler(num_blocks=10, max_num_batched_tokens=8)
    decoding, chunked = make_seq(8, max_tokens=20), make_seq(28, max_tokens=4)
    sched.add_sequence(decoding)
    sched.add_sequence(chunked)
    run(sched, zos=True)
    assert sched.num_preempted > 0
    assert [len(s.output_token_ids) for s in (decoding, chunked)] == [20, 4]
    assert sched.block_manager.num_free_blocks == sched.block_manager.num_usable_blocks


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="at least one token"):
        make_seq(0)
