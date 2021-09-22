#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
input = np.load('detector_output_without_deduplication.npy', allow_pickle=True).item()
block_size = input.get('block_size')
unique_blocks = len(input.get('list_blocks'))
tensor_shapes = input.get('blocked_tensor_dimension')
tensor_mapping = input.get('tensor_mapping')
num_tensors = len(tensor_shapes)

list_of_tensors = list()
for i in range (num_tensors):
    tensor_shapes[i] = input.get('blocked_tensor_dimension')[i]
for t in range(num_tensors):
    first, snd = tensor_shapes[t]
    l = list()
    for i in range(first):
        for j in range(snd):
            l.append(tensor_mapping[t].get((i,j)))
    list_of_tensors.append(l)
#print(list_of_tensors)


# In[ ]:


import numpy as np
from numpy.lib.arraysetops import unique
from bin_pack import *
import uuid
import hashlib
import timeit

def baseline(T, unique_blocks, l):
    I = set()
    for t_i in T:
        I = I.union(t_i)
    I = list(I)

    tensor_page_mapping = dict()
    tensor_page_set = set()

    i, j = 0, 0
    p_i_j = BinPackingScheme(I, l)

    # Process at all items in t0
    for r in range(1, len(T) + 1):
        for i in range(1, unique_blocks + 1):
            j = I.index(unique_blocks-1)
            s = math.ceil(j / l)
            tensor_page_set.add(s-1)
            p_i_j.mark(j, s)
        
        numBins = math.ceil( unique_blocks / l)
        p_i_j.numBins = numBins
        tensor_page_mapping[r] = tensor_page_set

    return set([p_i_j]), tensor_page_mapping


# In[ ]:


blocks_in_page = 10 # page can have 10 blocks
P = set()
start = timeit.default_timer()
P, tensor_page_mapping = baseline(list_of_tensors, unique_blocks, blocks_in_page)
stop = timeit.default_timer()
print('Time: ', stop - start) 
L = list(P)
print(L[0].numBins)
block_page_list = L[0].p_i_j
print(block_page_list)


# In[ ]:




