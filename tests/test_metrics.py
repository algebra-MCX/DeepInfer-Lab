"""CPU-only tests of request timestamps, latency metrics and goodput."""

import pytest

from deepinfer_lab.metrics import RequestMetrics, summarize
from deepinfer_lab.sampling_params import SamplingParams
from deepinfer_lab.sequence import Sequence, SequenceStatus
from tests.test_scheduler import make_scheduler, make_seq, run


def timed_seq(arrival, first, finish, num_output):
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=max(num_output, 1), seq_seed=0), 4)
    seq.token_ids += [7] * num_output
    seq.arrival_time, seq.first_token_time, seq.finish_time = arrival, first, finish
    return seq


def test_request_metrics_from_timestamps():
    m = RequestMetrics.from_sequence(timed_seq(10.0, 10.5, 11.5, 11))
    assert m.ttft == pytest.approx(0.5)
    assert m.tpot == pytest.approx(0.1)  # 10 gaps after the first token
    assert m.latency == pytest.approx(1.5)
    assert RequestMetrics.from_sequence(timed_seq(0.0, 1.0, 1.0, 1)).tpot == 0.0
    assert RequestMetrics.from_sequence(timed_seq(0.0, None, 1.0, 0)) is None


def test_goodput_counts_requests_meeting_both_slos():
    metrics = [
        RequestMetrics(ttft=1.0, tpot=0.05, latency=2.0, num_output_tokens=10),  # good
        RequestMetrics(ttft=2.0, tpot=0.10, latency=3.0, num_output_tokens=10),  # exactly at both SLOs: good
        RequestMetrics(ttft=3.0, tpot=0.05, latency=4.0, num_output_tokens=10),  # TTFT miss
        RequestMetrics(ttft=1.0, tpot=0.20, latency=4.0, num_output_tokens=10),  # TPOT miss
        None,  # never produced a token: excluded
    ]
    s = summarize(metrics, elapsed=2.0, ttft_slo=2.0, tpot_slo=0.1)
    assert s["requests"] == 4
    assert s["goodput"] == pytest.approx(1.0)
    assert s["slo_attainment"] == pytest.approx(0.5)
    assert s["output_throughput"] == pytest.approx(20.0)
    with pytest.raises(ValueError):
        summarize([None], elapsed=1.0, ttft_slo=1.0, tpot_slo=1.0)


@pytest.mark.parametrize("zos", [False, True])
def test_scheduler_stamps_first_token_and_finish_once(zos):
    sched = make_scheduler(num_blocks=6, max_num_seqs=4)  # small cache: forces preemption
    seqs = [make_seq(8, max_tokens=12) for _ in range(4)]
    for seq in seqs:
        sched.add_sequence(seq)
    run(sched, zos)
    assert sched.num_preempted > 0
    for seq in seqs:
        assert seq.status == SequenceStatus.FINISHED_LENGTH
        assert seq.arrival_time <= seq.first_token_time <= seq.finish_time


def test_abort_stamps_finish_without_first_token():
    sched = make_scheduler()
    seq = make_seq(8)
    sched.add_sequence(seq)
    assert sched.abort_request(seq.request_id) is seq
    assert seq.first_token_time is None and seq.finish_time is not None
    assert RequestMetrics.from_sequence(seq) is None
