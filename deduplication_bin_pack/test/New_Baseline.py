#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
input = np.load('detector_output.npy', allow_pickle=True).item()
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


from array import array
import math
from bin_pack import *

from numpy.lib.arraysetops import isin
from sympy.utilities.iterables import multiset_permutations
import hashlib
import numpy as np

def order_tensors_by_small_size(T):
    return sorted(T, key=lambda x: len(x), reverse=False)

def bin_pack_base(T, l):
    I = set()
    for t_i in T:
        I = I.union(t_i)
    I = list(I)
    
    items = T[0]

    i, j = 0, 0
    p_i_j = BinPackingScheme(I, l)

    # Process at all items in t0
    for i in range(1, len(items) + 1):
        # Use 1-index according to logic
        j = I.index(items[i - 1]) + 1
        s = math.ceil(j / l)
        p_i_j.mark(j, s)

    numBins = math.ceil(len(items) / l)

    # Already added tensor t1
    for i in range(2, len(T) + 1):
        bin_set, used_bin = p_i_j.findMinBinsMaxCover(T[i - 1],l)
        I_delta = set(T[i - 1]) - bin_set
        #print("I_delta")
        #print(I_delta)
        I_delta = list(I_delta)

        if not I_delta:
            continue
        else:
            remaining_items = order_tensor_blocks_by_freq(T, I_delta)
            #print(remaining_items)
            for j in range(1, len(remaining_items) + 1):
                # Important to index using I because we built BinPackingScheme using ordering of blocks in I
                s = I.index(remaining_items[j - 1]) + 1
                u = numBins + math.ceil(j / l)
                p_i_j.mark(s, u)

            numBins = numBins + math.ceil(len(remaining_items) / l)
            #print(numBins)
            p_i_j.numBins = numBins

    return set([p_i_j])


# In[ ]:


import timeit
import numpy as np

blocks_in_page = 10 # page can have 10 blocks
P = set()
list_of_tensors = order_tensors_by_small_size(list_of_tensors)
start = timeit.default_timer()
#P, tensor_page_mapping = bin_pack_greedy(list_of_tensors, blocks_in_page)
P = bin_pack_base(list_of_tensors, blocks_in_page)
stop = timeit.default_timer()
print('Time: ', stop - start) 
L = list(P)
print(L[0].numBins)
block_page_list = L[0].p_i_j
#print(block_page_list)


# In[ ]:




