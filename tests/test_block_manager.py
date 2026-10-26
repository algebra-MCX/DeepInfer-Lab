import pytest

from deepinfer_lab.engine.block_manager import AllocStatus, BlockManager


def test_block_zero_is_reserved():
    bm = BlockManager(5)
    assert bm.num_free_blocks == 4
    blocks = bm.allocate(4)
    assert sorted(blocks) == [1, 2, 3, 4]
    assert bm.num_free_blocks == 0


def test_alloc_status():
    bm = BlockManager(5)
    assert bm.can_allocate(4) == AllocStatus.OK
    assert bm.can_allocate(5) == AllocStatus.NEVER
    bm.allocate(2)
    assert bm.can_allocate(3) == AllocStatus.LATER


def test_free_order_and_reuse():
    bm = BlockManager(6)
    a = bm.allocate(2)  # [1, 2]
    b = bm.allocate(2)  # [3, 4]
    bm.free(a)
    # Block 5 was never used and is reused first; freed blocks are appended to the tail in reverse.
    assert bm.allocate(3) == [5, 2, 1]
    bm.free(b)
    assert bm.num_free_blocks == 2


def test_pin_ref_count():
    bm = BlockManager(4)
    blocks = bm.allocate(2)
    bm.pin(blocks)
    bm.free(blocks)
    assert bm.num_free_blocks == 1
    bm.free(blocks)
    assert bm.num_free_blocks == 3
    with pytest.raises(AssertionError):
        bm.free(blocks)
