"""KV cache block allocator, mirroring DeepInfer's csrc/engine/block_manager.hpp.

- Block 0 is reserved and never handed out: a 0 entry in a block table means "no block" (DeepInfer
  also points CUDA-graph padding tokens at it).
- Free blocks live in a doubly linked list. Allocation pops from the head and release pushes
  to the tail, so recently freed blocks are reused as late as possible.
- Blocks are reference counted so they can be pinned and shared (DeepInfer uses this for
  forked requests); a block returns to the free list when its last reference is dropped.
"""

from __future__ import annotations

import enum


class AllocStatus(enum.Enum):
    OK = enum.auto()  # Can be allocated now.
    LATER = enum.auto()  # Not enough free blocks now; may succeed after others are released.
    NEVER = enum.auto()  # Exceeds total cache capacity.


class BlockManager:
    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 1:
            raise ValueError(f"too few GPU blocks: {num_blocks}")
        self.num_blocks = num_blocks
        self.ref_count = [0] * num_blocks
        # Free list indexed by block id; 0 doubles as the null sentinel.
        self.prev = [0] * num_blocks
        self.next = [0] * num_blocks
        for i in range(1, num_blocks):
            self.prev[i] = i - 1
            self.next[i] = i + 1
        self.next[num_blocks - 1] = 0
        self.head, self.tail = 1, num_blocks - 1
        self.num_free_blocks = num_blocks - 1

    # Free list primitives.

    def _push_back(self, i: int) -> None:
        assert self.ref_count[i] == 0 and self.prev[i] == 0 and self.next[i] == 0
        self.prev[i] = self.tail
        if self.tail:
            self.next[self.tail] = i
        else:
            self.head = i
        self.tail = i
        self.num_free_blocks += 1

    def _pop_front(self) -> int:
        i = self.head
        assert i != 0, "free list is empty"
        self.head = self.next[i]
        if self.head:
            self.prev[self.head] = 0
        else:
            self.tail = 0
        self.next[i] = 0
        self.num_free_blocks -= 1
        return i

    # Public API.

    @property
    def num_usable_blocks(self) -> int:
        return self.num_blocks - 1

    def usage(self) -> float:
        return 1.0 - self.num_free_blocks / self.num_usable_blocks

    def can_allocate(self, n: int) -> AllocStatus:
        if n > self.num_usable_blocks:
            return AllocStatus.NEVER
        return AllocStatus.OK if self.num_free_blocks >= n else AllocStatus.LATER

    def allocate(self, n: int) -> list[int]:
        assert self.num_free_blocks >= n, f"need {n} blocks, only {self.num_free_blocks} free"
        blocks = []
        for _ in range(n):
            i = self._pop_front()
            self.ref_count[i] = 1
            blocks.append(i)
        return blocks

    def pin(self, block_ids: list[int]) -> None:
        for i in block_ids:
            assert 0 < i < self.num_blocks and self.ref_count[i] > 0, f"try to pin free block {i}"
            self.ref_count[i] += 1

    def free(self, block_ids: list[int]) -> None:
        # Release in reverse so a sequence's tail blocks are reused before its prefix blocks.
        for i in reversed(block_ids):
            assert 0 < i < self.num_blocks
            self.ref_count[i] -= 1
            assert self.ref_count[i] >= 0, f"ref_count of block {i} < 0"
            if self.ref_count[i] == 0:
                self._push_back(i)
