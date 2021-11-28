#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#input = np.load('detector_output.npy', allow_pickle=True).item()
#input = np.load('detector_output_diff_size_unshared_located_random.npy', allow_pickle=True).item()
#input = np.load('detector_output_same_size_unshared_located_random.npy', allow_pickle=True).item()
#input = np.load('detector_output_same_size_unshared_located_at_last.npy', allow_pickle=True).item()

list_of_tensors = list()

input_1 = np.load('civil_trainable_more_shareable_blocks.npy', allow_pickle=True).item()
block_size_1 = input_1.get('block_size')
#unique_blocks_1 = len(input_1.get('list_blocks'))
tensor_shapes_1 = input_1.get('blocked_tensor_dimension')
tensor_mapping_1 = input_1.get('tensor_mapping')
num_tensors_1 = len(tensor_shapes_1)

for i in range (num_tensors_1):
    tensor_shapes_1[i] = input_1.get('blocked_tensor_dimension')[i]
for t in range(num_tensors_1):
    first, snd = tensor_shapes_1[t]
    l = list()
    for i in range(first):
        for j in range(snd):
            l.append(tensor_mapping_1[t].get((i,j)))
    list_of_tensors.append(l)

input_2 = np.load('imdb_trainable.npy', allow_pickle=True).item()
block_size_2 = input_2.get('block_size')
#unique_blocks_2 = len(input_2.get('list_blocks'))
tensor_shapes_2 = input_2.get('blocked_tensor_dimension')
tensor_mapping_2 = input_2.get('tensor_mapping')
num_tensors_2 = len(tensor_shapes_2)

for i in range (num_tensors_2):
    tensor_shapes_2[i] = input_2.get('blocked_tensor_dimension')[i]
for t in range(num_tensors_2):
    first, snd = tensor_shapes_2[t]
    l = list()
    for i in range(first):
        for j in range(snd):
            l.append(tensor_mapping_2[t].get((i,j)))
    list_of_tensors.append(l)

input_3 = np.load('imdb_nontrainable.npy', allow_pickle=True).item()
block_size_3 = input_3.get('block_size')
#unique_blocks_3 = len(input_3.get('list_blocks'))
tensor_shapes_3 = input_3.get('blocked_tensor_dimension')
tensor_mapping_3 = input_3.get('tensor_mapping')
num_tensors_3 = len(tensor_shapes_3)

for i in range (num_tensors_3):
    tensor_shapes_3[i] = input_3.get('blocked_tensor_dimension')[i]
for t in range(num_tensors_3):
    first, snd = tensor_shapes_3[t]
    l = list()
    for i in range(first):
        for j in range(snd):
            l.append(tensor_mapping_3[t].get((i,j)))
    list_of_tensors.append(l)

input_4 = np.load('yelp_trainable.npy', allow_pickle=True).item()
block_size_4 = input_4.get('block_size')
#unique_blocks_4 = len(input_4.get('list_blocks'))
tensor_shapes_4 = input_4.get('blocked_tensor_dimension')
tensor_mapping_4 = input_4.get('tensor_mapping')
num_tensors_4 = len(tensor_shapes_4)

for i in range (num_tensors_4):
    tensor_shapes_4[i] = input_4.get('blocked_tensor_dimension')[i]
for t in range(num_tensors_4):
    first, snd = tensor_shapes_4[t]
    l = list()
    for i in range(first):
        for j in range(snd):
            l.append(tensor_mapping_4[t].get((i,j)))
    list_of_tensors.append(l)

input_5 = np.load('yelp_nontrainable.npy', allow_pickle=True).item()
block_size_5 = input_5.get('block_size')
tensor_shapes_5 = input_5.get('blocked_tensor_dimension')
tensor_mapping_5 = input_5.get('tensor_mapping')
num_tensors_5 = len(tensor_shapes_5)

for i in range (num_tensors_5):
    tensor_shapes_5[i] = input_5.get('blocked_tensor_dimension')[i]
for t in range(num_tensors_5):
    first, snd = tensor_shapes_5[t]
    l = list()
    for i in range(first):
        for j in range(snd):
            l.append(tensor_mapping_5[t].get((i,j)))
    list_of_tensors.append(l)
    
num_tensors = num_tensors_1 + num_tensors_2 + num_tensors_3 + num_tensors_4 + num_tensors_5

#print(list_of_tensors)
#print(num_tensors)


# In[2]:


from array import array
import math
from bin_pack import *

from numpy.lib.arraysetops import isin
from sympy.utilities.iterables import multiset_permutations
import hashlib
import numpy as np

#def order_tensors_by_small_size(T):
    #return sorted(T, key=lambda x: len(x), reverse=False)

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
        s = math.ceil(i / l)
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


# In[3]:


import timeit
import numpy as np

blocks_in_page = 5 # page can have 10 blocks
P = set()
#list_of_tensors = order_tensors_by_small_size(list_of_tensors)
start = timeit.default_timer()
#P, tensor_page_mapping = bin_pack_greedy(list_of_tensors, blocks_in_page)
P = bin_pack_base(list_of_tensors, blocks_in_page)
stop = timeit.default_timer()
print('Time: ', stop - start) 
L = list(P)
print(L[0].numBins)
#block_page_list = L[0].p_i_j
#print(block_page_list)


# In[ ]:




