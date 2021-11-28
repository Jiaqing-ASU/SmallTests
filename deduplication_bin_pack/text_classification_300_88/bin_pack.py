from array import array
import math

from numpy.lib.arraysetops import isin
from sympy.utilities.iterables import multiset_permutations
from sympy.utilities.iterables import multiset_combinations
import itertools
from math import factorial
import hashlib
import numpy as np
import gc

# Python program to demonstrate
# stack implementation using a linked list.
# node class
class Node:
   def __init__(self, value):
      self.value = value
      self.next = None
 
class Stack:
   # Initializing a stack.
   # Use a dummy node, which is
   # easier for handling edge cases.
   def __init__(self):
      self.head = Node("head")
      self.size = 0
 
   # String representation of the stack
   def __str__(self):
      cur = self.head.next
      out = ""
      while cur:
         out += str(cur.value) + "->"
         cur = cur.next
      return out[:-3]  
 
   # Get the current size of the stack
   def getSize(self):
      return self.size
    
   # Check if the stack is empty
   def isEmpty(self):
      return self.size == 0
    
   # Get the top item of the stack
   def peek(self):
       
      # Sanitary check to see if we
      # are peeking an empty stack.
      if self.isEmpty():
         raise Exception("Peeking from an empty stack")
      return self.head.next.value
 
   # Push a value into the stack.
   def push(self, value):
      node = Node(value)
      node.next = self.head.next
      self.head.next = node
      self.size += 1
      
   # Remove a value from the stack and return.
   def pop(self):
      if self.isEmpty():
         raise Exception("Popping from an empty stack")
      remove = self.head.next
      self.head.next = self.head.next.next
      self.size -= 1
      return remove.value

class BinPackingScheme(object):
    def __init__(self, item_ids, l):
        # Each row is a tensor
        # Each col is the bin/page
        self.p_i_j = [[0 for _ in range(math.ceil(len(item_ids) / l))] for _ in range(len(item_ids))]
        self.item_ids = list(item_ids)
        self.l = l
        self.numBins = math.ceil(len(item_ids) / l)

    def is_marked(self, item_id):
        return any([x == 1 for x in self.p_i_j[self.item_ids.index(item_id)]])

    def __eq__(self, other):
        my_array_hash = "".join([str(j) for sub in self.p_i_j for j in sub])
        other_array_hash = "".join([str(j) for sub in other.p_i_j for j in sub])
        if my_array_hash != other_array_hash:
            return False
        
        if len(self.item_ids) != len(other.item_ids):
            return False

        if self.numBins != other.numBins:
            return False

        # Order of items is also important
        for i in range(len(self.item_ids)):
            if self.item_ids[i] != other.item_ids[i]:
                return False

        return True

    def __ne__(self, other):
        my_array_hash = "".join([str(j) for sub in self.p_i_j for j in sub])
        other_array_hash = "".join([str(j) for sub in other.p_i_j for j in sub])
        if my_array_hash != other_array_hash:
            return True
        
        if len(self.item_ids) != len(other.item_ids):
            return True

        if self.numBins != other.numBins:
            return True

        # Order of items is also important
        for i in range(len(self.item_ids)):
            if self.item_ids[i] != other.item_ids[i]:
                return True

        return False

    def __hash__(self):
        """
        This is important. If this does not work, we cannot tell the difference between two bin packing schemes.
        What identifies a bin packing scheme is:
        1. The items being packed, i.e. the items must be uniquely identifiable
        2. The sequAmount of space savedence of packing pages into bins 
        """
        array_hash = "".join([str(j) for sub in self.p_i_j for j in sub])
        items_hash = "".join([str(hash(i)) for i in self.item_ids])
        full_hash = (array_hash + items_hash).encode("utf-8")

        return int(hashlib.md5(full_hash).hexdigest(), 16)

    def mark(self, i, j):
        if j - 1 > len(self.p_i_j[0]) - 1:
            diff = (j - 1) - (len(self.p_i_j[0]) - 1)
            # Add new bucket
            for row in self.p_i_j:
                row.extend([0 for _ in range(diff)])

        # Convert 1-index to 0-index
        self.p_i_j[i - 1][j - 1] = 1

    def merge(self, otherBinPackingScheme):
        assert self.l == otherBinPackingScheme.l

        for i in range(len(self.item_ids)):
            self.p_i_j[i] = self.p_i_j[i] + [0 for _ in range(otherBinPackingScheme.numBins)]

        # Take care of common item ids
        common_items = set(self.item_ids).intersection(set(otherBinPackingScheme.item_ids))
        for item in common_items:
            our_index = self.item_ids.index(item)
            their_index = otherBinPackingScheme.item_ids.index(item)
            self.p_i_j[our_index] += otherBinPackingScheme.p_i_j[their_index]

        # Take care of new item ids
        our_index = len(self.item_ids) - 1
        new_items = []
        for other_index, item in enumerate(otherBinPackingScheme.item_ids):
            if item in common_items:
                continue

            our_index += 1
            new_items.append(item)
            new_row = [0 for _ in range(self.numBins)] + otherBinPackingScheme.p_i_j[other_index]
            self.p_i_j.append(new_row)

        self.numBins += otherBinPackingScheme.numBins
        self.item_ids.extend(new_items)

        return self

    def blocks_in_bin_id(self, bin_id):
        return [self.item_ids[k] for k in range(len(self.item_ids)) if self.p_i_j[k][bin_id] == 1]

    """
    INPUT1: all_bins (a set of bins, each bin representing a page of tensor blocks)
    INPUT2: t (a set of tensor blocks)
    OUTPUT: bin_set (a minimum set of bins that maximally cover t)
    """
    def findMinBinsMaxCover(self, t, l):
        # A set of item ids
        T = set(t)
        # will contain groups of item ids
        bin_set = set()
        used_bins = set()
        #used_bin_set = set()

        while T:
            #cover = 0
            cover = l
            bin = None
            bin_id = None
            for j in range(self.numBins):
                if j in used_bins:
                    continue

                # Intersect tensor items in T with the items present in bin j
                bin_items = frozenset(self.blocks_in_bin_id(j))
                new_cover = len(T.intersection(bin_items))
                #if new_cover > cover:
                if new_cover == cover:
                    cover = new_cover
                    bin = bin_items
                    bin_id = j

            # If we have bins but their contents dont cover t at all i.e. different items
            if not bin:
                break

            used_bins.add(bin_id)
            #print("used_bins")
            #print(used_bins)
            bin_set = bin_set.union(T.intersection(bin))
            #used_bin_set = bin_set.union(T.intersection(used_bins))
            #print("bin_set")
            #print(bin_set)
            T = T - bin

            # All bins used
            if len(used_bins) == self.numBins:
                break

        return bin_set, used_bins
        #return used_bins

def remove_a_from_b(list_a,list_b):
    return [val for val in list_b if val not in list_a]

def itera(orderedList, all_combinations, current_combinations,l):
    if len(orderedList) != 0:
        # a set of possibile combination by selecting n from orderedList
        list_combinations_of_n = list(multiset_combinations(orderedList, min(l,len(orderedList))))
        #print(list_combinations_of_n)
        for seleted_items in list_combinations_of_n:
            orderedList_copy = orderedList.copy()
            orderedList_copy = remove_a_from_b(seleted_items, orderedList_copy)
            current_combinations_copy = current_combinations.copy()
            current_combinations_copy.extend(seleted_items)
            itera(orderedList_copy, all_combinations, current_combinations_copy,l)

    elif current_combinations != []:
        all_combinations.append(current_combinations)

# Compute the total number of unique k-combinations in a set of n elements.
# There are more efficient implementations of the choose function, but
# that's not the main point of this snippet.
def choose(n, k):
    if n < k:
        return 0
    return factorial(n) / (factorial(k) * factorial(n - k))

# Compute the mth combination in lexicographical order from a set of n
# elements chosen k at a time.
def combination(n, k, m):
    result = []
    a      = n
    b      = k
    x      = (choose(n, k) - 1) - m
    for i in range(0, k):
        a = a - 1
        while choose(a, b) > x:
            a = a - 1
        result.append(n - 1 - a)
        x = x - choose(a, b)
        b = b - 1
    return result

def itera_stack(orderedList, all_combinations, l):
    wait_for = Stack()
    wait_for.push(list(orderedList))
    be_processed = Stack()
    ini_list = list()
    be_processed.push(ini_list)

    while(wait_for.isEmpty() == False):
        current_list = wait_for.pop()
        current_processed = be_processed.pop()
        #print(current_list,current_processed)

        #list_combinations_of_n = list()
        #for i in range(min(l,len(current_list)), len(current_list)+1):
           #this_list = list(multiset_combinations(current_list, i))
           #for j in range(len(this_list)):
                #list_combinations_of_n.append(this_list[j])
        list_combinations_of_n = list(multiset_combinations(current_list, min(l,len(current_list))))


        print('start')
        print(list_combinations_of_n)
        print('end')
        for seleted_items in list_combinations_of_n:
            # print(type(seleted_items))
            new_wait = list(set(current_list) - set(seleted_items))
            new_processed = current_processed.copy()
            new_processed.extend(seleted_items)
            #print(new_wait, new_processed)
            if(len(new_wait) != 0):
                wait_for.push(new_wait)
                be_processed.push(new_processed)
            else:
                all_combinations.append(new_processed)

def itera_new(orderedList, all_combinations, l):
    this_list = list(multiset_combinations(orderedList, min(l,len(orderedList))))
    for i in range(len(this_list)):
        all_combinations.append(this_list[i])

def pack(t, org_t, l, list_of_real_len, all_combinations):
    #I = set(t)
    #print('t=',t)
    #I = list(I)
    #print('org_t',org_t)
    L = org_t
    I = t
    P = set()
    orderedList = I
    k = 0
    p_k = []


    min_combination = l - 1
    for i in range(len(list_of_real_len)):
        if((list_of_real_len[i]%l) < min_combination):
            min_combination = list_of_real_len[i]%l

    #print(min_combination)

    #nextPermutation = list()
    seen_perm = {
        frozenset(frozenset(org_t[i:i + l]) for i in range(0, len(org_t), l)): True
    }

    #all_combinations = list()
    #current_combinations = list()
    #itera(orderedList, all_combinations, current_combinations,l)





    for i in range(min_combination, min(len(orderedList)+1,l+1)):
        itera_new(orderedList, all_combinations, i)

    print('all_combinations',len(all_combinations))

    #print(all_combinations)
    nextPermutation = ( y for y in all_combinations)

    #print(org_t)
    perm = 0

    while org_t:
        #print(f"Permutation {perm}")
        #perm += 1
        next_tuple_list = list()
        curr_com = all_combinations[perm]
        n = 0
        for i in range(len(curr_com)):
            if (i == 0):
                mytuple = (curr_com[i],0)
                next_tuple_list.append(mytuple)
            else:
                if(curr_com[i] != curr_com[i - 1]):
                    n = 0
                    mytuple = (curr_com[i],n)
                    next_tuple_list.append(mytuple)
                else:
                    n = n + 1
                    mytuple = (curr_com[i],n)
                    next_tuple_list.append(mytuple)

        #print('next_tuple_list', next_tuple_list)
        i, j = 0, 0
        p_i_j = BinPackingScheme(L, l)

        for item in range(len(next_tuple_list)):
            #print('Item=', item)
            # Use 1-index according to logic
            #i = I.index(item)
            i = L.index(next_tuple_list[item])
            #print(i)
            #j = math.ceil(orderedList.index(item) / l)
            #j = math.ceil(item / l)
            j = int(item / l) + 1
            #j = 1
            #print(j)
            p_i_j.mark(i, j)


            #j = I.index(items[i - 1]) + 1
            ##s = math.ceil(j / l)
            #s = math.ceil(i / l)

        remaining_list = list(set(L)-set(next_tuple_list))
        #print('remaining_list',remaining_list)
        for item in range(len(remaining_list)):
            #print('Item=', item)
            # Use 1-index according to logic
            #i = I.index(item)
            i = L.index(remaining_list[item])
            #print(i)
            #j = math.ceil(orderedList.index(item) / l)
            #j = math.ceil(item / l) + 1
            j = int(item / l) + 2
            #j = 2
            #print(j)
            p_i_j.mark(i, j)

        p_k.append(p_i_j)
        P = P.union(set([p_k[k]]))
        k = k + 1

        try:
            key = frozenset(frozenset(org_t[i:i + l]) for i in range(0, len(org_t), l))
            #items = 0
            while seen_perm.get(key, False):
                # print(f"Skipping")
                del key
                gc.collect()
                org_t = next(nextPermutation)
                perm = perm + 1
                #orderedList = nextPermutation[items]
                #items += 1
                #print("orderedList:", orderedList)
                key = frozenset(frozenset(org_t[i:i + l]) for i in range(0, len(org_t), l))
            seen_perm[key] = True
        except StopIteration:
            org_t = None


    #perm = 0
    '''
    while orderedList:
        #print(f"Permutation {perm}")
        #perm += 1

        i, j = 0, 0
        p_i_j = BinPackingScheme(orderedList, l)
        for item in range(len(orderedList)):
            #print('Item=', item)
            # Use 1-index according to logic
            #i = I.index(item)
            i = I.index(orderedList[item-1])
            #print(i)
            #j = math.ceil(orderedList.index(item) / l)
            j = math.ceil(item / l)
            p_i_j.mark(i, j)


            #j = I.index(items[i - 1]) + 1
            ##s = math.ceil(j / l)
            #s = math.ceil(i / l)

        p_k.append(p_i_j)
        P = P.union(set([p_k[k]]))
        k = k + 1

        try:
            key = frozenset(frozenset(orderedList[i:i + l]) for i in range(0, len(orderedList), l))
            #items = 0
            while seen_perm.get(key, False):
                # print(f"Skipping")
                del key
                gc.collect()
                orderedList = next(nextPermutation)
                #orderedList = nextPermutation[items]
                #items += 1
                #print("orderedList:", orderedList)
                key = frozenset(frozenset(orderedList[i:i + l]) for i in range(0, len(orderedList), l))
            seen_perm[key] = True
        except StopIteration:
            orderedList = None
    '''

    # A set of BinPackingScheme's, each BinPackingScheme is a 2-D array -
    # If Pij = 0, it means block i is not in page j;
    # If Pij = 1, it means block i is in page j.
    return P

def pack_for_adjust(t, l):
    I = set(t)
    #print('t=',t)
    I = list(I)

    P = set()
    orderedList = I
    k = 0
    p_k = []

    #nextPermutation = list()
    #seen_perm = {
    #    frozenset(frozenset(orderedList[i:i + l]) for i in range(0, len(orderedList), l)): True
    #}

    #all_combinations = list()
    #current_combinations = list()
    #itera(orderedList, all_combinations, current_combinations,l)
    #itera_stack(orderedList, all_combinations, l)
    #print(all_combinations)
    #nextPermutation = ( y for y in all_combinations)
    
    #perm = 0
    #while orderedList:
        #print(f"Permutation {perm}")
        #perm += 1

    i, j = 0, 0
    p_i_j = BinPackingScheme(orderedList, l)
    for item in range(len(orderedList)):
        i = I.index(orderedList[item-1])
        #print(i)
        #j = math.ceil(orderedList.index(item) / l)
        j = math.ceil(item / l) + 2
        p_i_j.mark(i, j)


            #j = I.index(items[i - 1]) + 1
            ##s = math.ceil(j / l)
            #s = math.ceil(i / l)

    p_k.append(p_i_j)
    P = P.union(set([p_k[k]]))
    k = k + 1

        #try:
        #    key = frozenset(frozenset(orderedList[i:i + l]) for i in range(0, len(orderedList), l))
            #items = 0
        #    while seen_perm.get(key, False):
                # print(f"Skipping")
        #        del key
        #        gc.collect()
        #        orderedList = next(nextPermutation)
                #orderedList = nextPermutation[items]
                #items += 1
                #print("orderedList:", orderedList)
        #        key = frozenset(frozenset(orderedList[i:i + l]) for i in range(0, len(orderedList), l))
        #    seen_perm[key] = True
        #except StopIteration:
        #    orderedList = None

    # A set of BinPackingScheme's, each BinPackingScheme is a 2-D array -
    # If Pij = 0, it means block i is not in page j;
    # If Pij = 1, it means block i is in page j.
    return P








"""
1: INPUT1: P_star (a previous list of optimal bin-packing schemes - a set of bin sets)
2: INPUT2: 𝑡 (a new tensor to be added - a set of tensor blocks)
3: INPUT3: 𝑙 (the maximum number of items for each bin)
4: OUTPUT: P = {𝑃𝑘 } (a list of optimal bin-packing schemes - a set of bin sets)
           A bin-packing scheme is a set of bins
"""

def adjust_greedy(P_star, t, org_t, l):

    # Again, t can have duplicates?
    #I = set(t)

    I = set(org_t)
    #I = t

    #I = t

    minNumBins = math.inf
    P = set()

    #print(len(P_star))

    # P_k is a BinPackingScheme which is a 2-D array -
    # If Pij = 0, it means block i is not in page j;
    # If Pij = 1, it means block i is in page j.
    for P_k in P_star:
        #print('P_k=',P_k.numBins)
        print(P_k.p_i_j)
        bin_set, used_bins = P_k.findMinBinsMaxCover(I,l)

        print('bin_set',bin_set)
        #tensor_page_set = used_bins
        I_delta = I - bin_set

        I_delta = list(I_delta)

        print('I_delta=',I_delta)

        if not I_delta:
            if P_k.numBins == minNumBins:
                P = P.union(set([P_k]))

            else:
                if P_k.numBins < minNumBins:
                    P = set([P_k])
                    minNumBins = P_k.numBins
        else:
            P_prime = pack_for_adjust(I_delta, l)
            for P_dash in P_prime:
                P_new = P_dash.merge(P_k)
                #P_new = P_dash.union(P_k)
                if P_new.numBins == minNumBins:
                    P = P.union(set([P_new]))
                else:
                    if P_new.numBins < minNumBins:
                        P = set([P_new])
                        minNumBins = P_new.numBins
    #print(P)
    #tensor_page_mapping[i] = tensor_page_set
    return P
    
def adjust(P_star, t, org_t, l, all_combinations):
    # Again, t can have duplicates?
    #I = set(t)

    I = set(org_t)

    minNumBins = math.inf
    P = set()
    #print(len(P_star))

    # P_k is a BinPackingScheme which is a 2-D array -
    # If Pij = 0, it means block i is not in page j;
    # If Pij = 1, it means block i is in page j.
    which_com = 0
    for P_k in P_star:

        next_tuple_list = list()
        curr_com = all_combinations[which_com]
        n = 0
        for i in range(len(curr_com)):
            if (i == 0):
                mytuple = (curr_com[i],0)
                next_tuple_list.append(mytuple)
            else:
                if(curr_com[i] != curr_com[i - 1]):
                    n = 0
                    mytuple = (curr_com[i],n)
                    next_tuple_list.append(mytuple)
                else:
                    n = n + 1
                    mytuple = (curr_com[i],n)
                    next_tuple_list.append(mytuple)

        #print('next_tuple_list', next_tuple_list)

        #print('P_k', P_k.p_i_j)
        #bin_set, used_bins= P_k.findMinBinsMaxCover(org_t,l)

        which_com = which_com + 1
        #I_delta = I - bin_set
        I_delta = I - set(next_tuple_list)

        #print('I_delta=',I_delta)
        I_delta = list(I_delta)
        #print('I_delta=',I_delta)

        if not I_delta:
            P = P.union(set([P_k]))
        else:
            P_prime = pack_for_adjust(I_delta, l)
            for P_dash in P_prime:
                P_new = P_dash.merge(P_k)
                P = P.union(set([P_new]))
    
    return P
 

def test_adjust():
    # Can have repeated elements
    t1 = [("t1", 0, 1), ("t1", 0, 0), ("t1", 0, 0), ("t1", 1, 1)]
    t2 = [("t1", 0, 1), ("t1", 0, 0), ("t2", 1, 1)]
    l = 2
    P_star = pack(t1, l)

    P = adjust(P_star, t2, l)


"""
INPUT1: T (A set of tensors, each tensor is a set of block ids) - each can have duplicates?
INPUT2: l (the maximum number of items that can be held in a bin)
OUTPUT: P={𝑃𝑖} (a list of optimal bin-packing schemes)
"""

def bin_pack_dp(T, org_T, l, list_of_real_len):
    initialized = False
    P_star = None
    tensor_page_mapping = dict()
    tensor_page_set = set()
    all_combinations = list()
    for i in range(len(T)):
        if not initialized:
            P_star = pack(T[i], org_T[i], l, list_of_real_len, all_combinations)
            initialized = True
            #print('P_star', len(P_star))
        else:
            P_star = adjust(P_star, T[i], org_T[i], l, all_combinations)

    return P_star

def bin_pack_dp_greedy(T, l):
    initialized = False
    P_star = None
    for t_i in T:
        if not initialized:
            P_star = pack(t_i, l)
            initialized = True
        else:
            P_star = adjust_greedy(P_star, t_i, l)

    return P_star


def order_tensors_by_size(T):
    return sorted(T, key=lambda x: len(x), reverse=True)

def order_tensors_by_size_small(T):
    return sorted(T, key=lambda x: len(x), reverse=False)

def order_tensor_blocks_by_freq(T, t_i):
    freq_map = {}
    for block in t_i:
        if not block in freq_map:
            freq_map[block] = 1
        else:
            freq_map[block] += 1

    for t_k in T:
        if not isinstance(t_i, list) and t_k == t_i:
            continue

        for block in t_k:
            if block in freq_map:
                freq_map[block] += 1

    #print(freq_map)

    ordered_items = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
    #print(ordered_items)
    #print ([x[0] for x in ordered_items])
    return [x[0] for x in ordered_items]

def len_unique_items(T):
    items = set()
    for t_i in T:
        items.union(t_i)

    return len(items)

"""
1:INPUT1: 𝑇 (a set of tensors, each tensor is a set of item ids i.e. tensor blocks ids)
2:INPUT2: 𝑙 (the maximum number of items for each bin)
3:OUTPUT: {𝑃𝑖𝑗} (an approximate optimal bin-packing scheme)
"""
def bin_pack_greedy(T, l):
    I = set()
    for t_i in T:
        I = I.union(t_i)
    I = list(I)

    tensor_page_mapping = dict()
    tensor_page_set = set()
    
    tensors = order_tensors_by_size(T)
    # Add tensor t1
    items = order_tensor_blocks_by_freq(T, tensors[0])
    #print(items)

    #print(I)
    #print(type(I))
    
    i, j = 0, 0
    p_i_j = BinPackingScheme(I, l)

    # Process at all items in t0
    for i in range(1, len(items) + 1):
        # Use 1-index according to logic
        j = I.index(items[i - 1]) + 1
        #s = math.ceil(j / l)
        s = math.ceil(i / l)
        tensor_page_set.add(s-1)
        #print('j=',j)
        #print('s=',s)
        p_i_j.mark(j, s)

    numBins = math.ceil(len(items) / l)
    p_i_j.numBins = numBins
    tensor_page_mapping[0] = tensor_page_set


    # Already added tensor t1
    for i in range(2, len(T) + 1):
        bin_set, used_bins = p_i_j.findMinBinsMaxCover(tensors[i - 1],l)
        tensor_page_set = used_bins
        #print("tensor_page_set")
        #print(tensor_page_set)
        I_delta = set(tensors[i - 1]) - bin_set
        I_delta = list(I_delta)

        if not I_delta:
            continue
        else:
            remaining_items = order_tensor_blocks_by_freq(T, I_delta)
            for j in range(1, len(remaining_items) + 1):
                # Important to index using I because we built BinPackingScheme using ordering of blocks in I
                s = I.index(remaining_items[j - 1]) + 1
                u = numBins + math.ceil(j / l)
                tensor_page_set.add(u-1)
                p_i_j.mark(s, u)

            numBins = numBins + math.ceil(len(remaining_items) / l)
            p_i_j.numBins = numBins
        tensor_page_mapping[i-1] = tensor_page_set

    return set([p_i_j]), tensor_page_mapping

def bin_pack_greedy_small2large(T, l):
    I = set()
    for t_i in T:
        I = I.union(t_i)
    I = list(I)

    tensor_page_mapping = dict()
    tensor_page_set = set()
    
    tensors = order_tensors_by_size_small(T)
    # Add tensor t1
    items = order_tensor_blocks_by_freq(T, tensors[0])
    #print(items)

    #print(I)
    #print(type(I))
    
    i, j = 0, 0
    p_i_j = BinPackingScheme(I, l)

    # Process at all items in t0
    for i in range(1, len(items) + 1):
        # Use 1-index according to logic
        j = I.index(items[i - 1]) + 1
        #s = math.ceil(j / l)
        s = math.ceil(i / l)
        tensor_page_set.add(s-1)
        #print('j=',j)
        #print('s=',s)
        p_i_j.mark(j, s)

    numBins = math.ceil(len(items) / l)
    p_i_j.numBins = numBins
    tensor_page_mapping[0] = tensor_page_set


    # Already added tensor t1
    for i in range(2, len(T) + 1):
        bin_set, used_bins = p_i_j.findMinBinsMaxCover(tensors[i - 1],l)
        tensor_page_set = used_bins
        #print("tensor_page_set")
        #print(tensor_page_set)
        I_delta = set(tensors[i - 1]) - bin_set
        I_delta = list(I_delta)

        if not I_delta:
            continue
        else:
            remaining_items = order_tensor_blocks_by_freq(T, I_delta)
            for j in range(1, len(remaining_items) + 1):
                # Important to index using I because we built BinPackingScheme using ordering of blocks in I
                s = I.index(remaining_items[j - 1]) + 1
                u = numBins + math.ceil(j / l)
                tensor_page_set.add(u-1)
                p_i_j.mark(s, u)

            numBins = numBins + math.ceil(len(remaining_items) / l)
            p_i_j.numBins = numBins
        tensor_page_mapping[i-1] = tensor_page_set

    return set([p_i_j]), tensor_page_mapping