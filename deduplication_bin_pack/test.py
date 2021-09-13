import numpy as np
from numpy.lib.arraysetops import unique
from bin_pack import *
import uuid
import hashlib


"""
10 tensors
each tensor has - 10^4 x 10^4 ~100MB - 10k blocks in total, 1k, 2k, 4k, 8k ... unique blocks
ratio of unique blocks in each tensor
1. evenly distribute
2. ~80% shared, 20% unshared for every tensor
3. Probablility distribution - random
Each unique block has 10% share in each tensor

4. For each unique block, sample a tensor, place it into that tensor. Repeat with every unique block till all tensors are filled 
5. For each tensor, sample a unique block

Report ideal deduplication factor - max number of blocks 
"""

# 1. Amount of space saved
# 2. How much time for dp vs greedy
# 3. Naive packing comparision - time and space i.e data loading time for non-shared


class Tensor(object):
    def __init__(self, blocks=None, shape=None, block_shape=None, name=None):
        self.tensor_blocks = blocks or []
        self.shape = shape
        self.block_shape = block_shape
        self.name = name

    def __len__(self):
        return len(self.tensor_blocks)

    def num_blocks(self):
        if not self.shape:
            return 0
        else:
            return np.multiply(*self.shape)

    def __eq__(self, o):
        return self.name == o.name

    def __ne__(self, o):
        return self.name != o.name

    def __hash__(self):
        return int(hashlib.md5(self.name.encode('utf-8')).hexdigest(), 16)

    def get_block(self):
        return (self.name,)

    def __getitem__(self, idx):
        return self.tensor_blocks[idx]

    def __setitem__(self, idx, item):
        self.tensor_blocks[idx] = item

    def __iter__(self):
        self.pos = 0
        return self

    def __next__(self):
        if self.pos >= len(self.tensor_blocks):
            raise StopIteration

        ret = self.tensor_blocks[self.pos]
        self.pos += 1
        return ret


# def equal_distribution(tensors, unique_blocks, use_all=True):
#     unique_per_tensor = unique_blocks // len(tensors)
#     exclude = []
#     left = 0
#     for t in tensors:
#         if t.num_blocks() < unique_per_tensor:
#             t.tensor_blocks = t.num_blocks()
#             left += unique_per_tensor - t.num_blocks()
#             exclude.append(t)
#         else:
#             t.tensor_blocks = unique_per_tensor

#     """
#     If tensor sizes are randomly generated, its possible that unique_per_tensor > the size of the tensor
#     Fill up the small tensors and redistribute the remaining unique blocks among the other remaining tensors
#     """
#     if left > 0 and exclude and use_all:
#         equal_distribution([t for t in tensors if t not in exclude], left)


# def percentage_distribution(tensors, percentage_unique_per_tensor):
#     # We may not use all the unique_blocks depending on the shapes of the arrays
#     for t in tensors:
#         t.tensor_blocks = percentage_unique_per_tensor * t.num_blocks()


def block_distribution(tensors, unique_blocks):
    T = [t for t in tensors]
    while T:
        for b in unique_blocks:
            t = np.random.choice(T, 1)[0]
            t.tensor_blocks.append(b)
            if len(t.tensor_blocks) == t.num_blocks():
                T.remove(t)


def tensor_distribution(tensors, unique_blocks):
    T = [t for t in tensors]
    while T:
        for t in tensors:
            b = np.random.choice(unique_blocks, 1)[0]
            t.tensor_blocks.append(b)
            if len(t.tensor_blocks) == t.num_blocks():
                T.remove(t)


distribution_mode = {
    # "equal": equal_distribution,
    # "percentage": percentage_distribution,
    "block": block_distribution,
    "tensor": tensor_distribution
}


def generate_random_tensors(
    num_tensors,
    block_shape,
    num_unique_blocks,
    distribution_ops,
    max_tensor_blocks=None,
    tensor_shape=None
):
    assert(num_tensors > 0)
    unique_blocks = [uuid.uuid1().hex for _ in range(num_unique_blocks)]

    tensor_shapes = []

    if max_tensor_blocks and not tensor_shape and isinstance(max_tensor_blocks, tuple):
        rng = np.random.default_rng(12345)
        tensor_shapes = [
            (rng.integers(low=1, high=max_tensor_blocks[0], size=1), rng.integers(low=1, high=max_tensor_blocks[1], size=1),)
            for _ in range(num_tensors)
        ]
    elif tensor_shape and not max_tensor_blocks:
        if isinstance(tensor_shape, tuple):
            tensor_shapes = [tensor_shape for _ in range(num_tensors)]
        elif isinstance(tensor_shape, list) and len(tensor_shape) == num_tensors and isinstance(tensor_shape[0], tuple):
            tensor_shapes = tensor_shape
        else:
            raise Exception("tensor_shape must be a tuple or a list of tuples")
    else:
        raise Exception("Either need tensor_shape for user-defined sized tensors or max_tensor_blocks tuple to generate random shaped tensors")

    total_blocks = np.prod([a * b for a,b in tensor_shapes])

    tensors = [Tensor(name=f"t{i}", shape=tensor_shapes[i], block_shape=block_shape) for i in range(num_tensors)]

    distribution_mode[distribution_ops['mode']](tensors, unique_blocks, **distribution_ops.get('kwargs', {}))

    return tensors, total_blocks


num_tensors = 5
block_size = (100, 100,)
unique_blocks = 100
tensor_shape = (5, 5,) # 25 blocks

T, total_blocks = generate_random_tensors(num_tensors, block_size, unique_blocks, {'mode': 'tensor'}, tensor_shape=tensor_shape)
l = 10 # page can have 10 blocks

P = bin_pack_greedy(T, l)
import pdb
pdb.set_trace()

"""
p_i_j
  p1 p2 p3 ..
1
2
3
4
.
.
"""