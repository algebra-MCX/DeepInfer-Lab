"""CPU-only LLMEngine tests; the model runner is replaced by a fake step handle."""

from minideepinfer.engine.block_manager import BlockManager
from minideepinfer.engine.llm_engine import LLMEngine, StatsLogger
from minideepinfer.engine.scheduler import Scheduler
from minideepinfer.sampling_params import SamplingParams
from minideepinfer.sequence import Sequence, SequenceStatus


class FakeHandle:
    def __init__(self, sched, token_ids):
        self.sched, self.token_ids = sched, token_ids

    def wait(self):
        return self.token_ids


def make_engine():
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = Scheduler(BlockManager(16), BlockManager(4), max_num_seqs=2, max_num_batched_tokens=32, max_model_len=64)
    engine.block_manager = engine.scheduler.block_manager
    engine.stats = StatsLogger(0.0)
    engine.inflight, engine.pending_finished, engine._requests = None, [], {}
    return engine


def test_abort_keeps_sequences_finished_by_the_drained_step():
    engine = make_engine()
    done = Sequence([1, 2, 3], SamplingParams(max_tokens=1, seq_seed=0), 4, request_id="done")
    other = Sequence([1, 2, 3], SamplingParams(max_tokens=4, seq_seed=0), 4, request_id="other")
    engine.scheduler.add_sequence(done)
    engine.scheduler.add_sequence(other)
    sched = engine.scheduler.schedule()
    engine.inflight = FakeHandle(sched, [7, 7])  # prefill in flight; `done` finishes once processed

    assert engine.abort_request("other") is other
    assert done.status == SequenceStatus.FINISHED_LENGTH
    assert engine.has_unfinished_requests()
    assert engine.step() == [done]
    assert not engine.has_unfinished_requests()
