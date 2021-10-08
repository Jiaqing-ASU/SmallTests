#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools

output = np.load('tensor_list.npy', allow_pickle=True).item()
#output.keys()
#print(output['list_of_tensors'])

list = [{}, {}, {}, {}, {}]


for num in range(5):
    a0=set(output['list_of_tensors'][0+num*5])
    a1=set(output['list_of_tensors'][1+num*5])
    a2=set(output['list_of_tensors'][2+num*5])
    a3=set(output['list_of_tensors'][3+num*5])
    a4=set(output['list_of_tensors'][4+num*5])
    list[num]=a0.union(a1, a2, a3, a4)
    #print(list[num])

#index = [0, 1, 2, 3, 4]
#for L in range(0, len(index)+1):
#    for subset in itertools.combinations(index, L):
#        print(subset)

print("Shared by all tensors:")
l01234=list[0].intersection(list[1], list[2], list[3], list[4])
#print(l01234)
print(len(l01234))

print("Shared by 0, 1, 2, 3:")
l0123=list[0].intersection(list[1], list[2], list[3])-l01234
#print(l0123)
print(len(l0123))

print("Shared by 0, 1, 2, 4:")
l0124=list[0].intersection(list[1], list[2], list[4])-l01234
#print(l0124)
print(len(l0124))

print("Shared by 0, 1, 3, 4:")
l0134=list[0].intersection(list[1], list[3], list[4])-l01234
#print(l0134)
print(len(l0134))

print("Shared by 0, 2, 3, 4:")
l0234=list[0].intersection(list[2], list[3], list[4])-l01234
#print(l0234)
print(len(l0234))

print("Shared by 1, 2, 3, 4:")
l1234=list[1].intersection(list[2], list[3], list[4])-l01234
#print(l1234)
print(len(l1234))

print("Shared by 0, 1, 2:")
l012=list[0].intersection(list[1], list[2])-l0123-l0124-l01234
#print(l012)
print(len(l012))

print("Shared by 0, 1, 3:")
l013=list[0].intersection(list[1], list[3])-l0123-l0134-l01234
#print(l013)
print(len(l013))

print("Shared by 0, 1, 4:")
l014=list[0].intersection(list[1], list[4])-l0124-l0134-l01234
#print(l014)
print(len(l014))

print("Shared by 0, 2, 3:")
l023=list[0].intersection(list[2], list[3])-l0123-l0234-l01234
#print(l023)
print(len(l023))

print("Shared by 0, 2, 4:")
l024=list[0].intersection(list[2], list[4])-l0124-l0234-l01234
#print(l024)
print(len(l024))

print("Shared by 0, 3, 4:")
l034=list[0].intersection(list[3], list[4])-l0134-l0234-l01234
#print(l034)
print(len(l034))

print("Shared by 1, 2, 3:")
l123=list[1].intersection(list[2], list[3])-l0123-l1234-l01234
#print(l123)
print((l123))

print("Shared by 1, 2, 4:")
l124=list[1].intersection(list[2], list[4])-l0124-l1234-l01234
#print(l124)
print(len(l124))

print("Shared by 1, 3, 4:")
l134=list[1].intersection(list[3], list[4])-l0134-l1234-l01234
#print(l134)
print(len(l134))

print("Shared by 2, 3, 4:")
l234=list[2].intersection(list[3], list[4])-l0234-l1234-l01234
#print(l234)
print(len(l234))

print("Shared by 0, 1:")
l01=list[0].intersection(list[1])-l012-l013-l014-l0123-l0124-l0134-l01234
#print(l01)
print(len(l01))

print("Shared by 0, 2:")
l02=list[0].intersection(list[2])-l012-l023-l024-l0123-l0124-l0234-l01234
#print(l02)
print(len(l02))

print("Shared by 0, 3:")
l03=list[0].intersection(list[3])-l013-l023-l034-l0123-l0234-l0234-l01234
#print(l03)
print(len(l03))

print("Shared by 0, 4:")
l04=list[0].intersection(list[4])-l014-l024-l034-l0124-l0134-l0234-l01234
#print(l04)
print(len(l04))

print("Shared by 1, 2:")
l12=list[1].intersection(list[2])-l012-l123-l124-l0123-l0124-l1234-l01234
#print(l12)
print(len(l12))

print("Shared by 1, 3:")
l13=list[1].intersection(list[3])-l013-l034-l134-l0123-l0134-l1234-l01234
#print(l13)
print(len(l13))

print("Shared by 1, 4:")
l14=list[1].intersection(list[4])-l014-l124-l134-l0124-l0134-l1234-l01234
#print(l14)
print(len(l14))

print("Shared by 2, 3:")
l23=list[2].intersection(list[3])-l023-l123-l234-l0123-l0234-l1234-l01234
#print(l23)
print(len(l23))

print("Shared by 2, 4:")
l24=list[2].intersection(list[4])-l024-l124-l234-l0124-l0234-l1234-l01234
#print(l24)
print(len(l24))

print("Shared by 3, 4:")
l34=list[3].intersection(list[4])-l034-l134-l234-l0134-l0234-l1234-l01234
#print(l34)
print(len(l34))

print("Private to 0:")
l0=list[0]-l01-l02-l03-l04-l012-l013-l014-l023-l024-l034-l0123-l0124-l0134-l0234-l01234
#print(l0)
print(len(l0))

print("Private to 1:")
l1=list[1]-l01-l12-l13-l14-l012-l013-l014-l123-l124-l134-l0123-l0124-l0134-l1234-l01234
#print(l1)
print(len(l1))

print("Private to 2:")
l2=list[2]-l02-l12-l23-l24-l012-l023-l024-l123-l124-l234-l0123-l0124-l0234-l1234-l01234
#print(l2)
print(len(l2))

print("Private to 3:")
l3=list[3]-l03-l13-l23-l34-l013-l023-l034-l123-l134-l234-l0123-l0134-l0234-l1234-l01234
#print(l3)
print(len(l3))

print("Private to 4:")
l4=list[4]-l04-l14-l24-l34-l014-l024-l034-l124-l134-l234-l0124-l0134-l0234-l1234-l01234
#print(l4)
print(len(l4))


# In[6]:


import numpy as np

ll0=[491,501,487,488,493,494,665,666,667,668,669,670,671]
ll1=[491,410,490,561,562]
ll2=[491,501,487,488,493,494,410,490,415,512,509,510,511]
ll3=[501,659,660,661,662]
ll4=[491,501,487,488,493,494,410,490,415,505,506,507,508]

list_of_tensors = [ll0,ll1,ll2,ll3,ll4]

print(list_of_tensors)


# In[ ]:


import numpy as np
from numpy.lib.arraysetops import unique
from bin_pack import *
import uuid
import hashlib
import timeit

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

blocks_in_page = 8 # page can have 10 blocks
P = set()
start = timeit.default_timer()
P = bin_pack_dp_greedy(list_of_tensors, blocks_in_page)

stop = timeit.default_timer()
print('Time: ', stop - start) 
L = list(P)
print(L[0].numBins)


# In[ ]:




