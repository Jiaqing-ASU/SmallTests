from array import array
import math

from numpy.lib.arraysetops import isin
from sympy.utilities.iterables import multiset_permutations
import hashlib
import numpy as np

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
    def findMinBinsMaxCover(self, t):
        # A set of item ids
        T = set(t)
        # will contain groups of item ids
        bin_set = set()
        used_bins = set()

        while T:
            cover = 0
            bin = None
            bin_id = None
            for j in range(self.numBins):
                if j in used_bins:
                    continue

                # Intersect tensor items in T with the items present in bin j
                bin_items = frozenset(self.blocks_in_bin_id(j))
                new_cover = len(T.intersection(bin_items))
                if new_cover > cover:
                    cover = new_cover
                    bin = bin_items
                    bin_id = j

            # If we have bins but their contents dont cover t at all i.e. different items
            if not bin:
                break

            used_bins.add(bin_id)
            bin_set = bin_set.union(T.intersection(bin))
            T = T - bin

            # All bins used
            if len(used_bins) == self.numBins:
                break

        return bin_set

"""
1: INPUT1: 𝐼 (the item ids included in the tensor)
2: INPUT2: 𝑙 (the maximum number of items for each bin)
3: OUTPUT: P = {𝑃𝑘 } (a list of optimal bin-packing schemes)
"""
def pack(t, l):
    I = set(t)
    I = list(I)

    P = set()
    orderedList = I
    k = 0
    p_k = []

    seen_perm = {
        frozenset(frozenset(orderedList[i:i + l]) for i in range(0, len(orderedList), l)): True
    }

    nextPermutation = multiset_permutations(orderedList)

    perm = 0
    while orderedList:
        print(f"Permutation {perm}")
        perm += 1

        i, j = 0, 0
        p_i_j = BinPackingScheme(orderedList, l)
        for item in orderedList:
            # Use 1-index according to logic
            i = I.index(item) + 1
            j = math.ceil(orderedList.index(item) / l)

            p_i_j.mark(i, j)

        p_k.append(p_i_j)
        P = P.union(set([p_k[k]]))
        k = k + 1

        try:
            key = frozenset(frozenset(orderedList[i:i + l]) for i in range(0, len(orderedList), l))
            while seen_perm.get(key, False):
                # print(f"Skipping")
                orderedList = next(nextPermutation)
                key = frozenset(frozenset(orderedList[i:i + l]) for i in range(0, len(orderedList), l))
            seen_perm[key] = True
        except StopIteration:
            orderedList = None

    # A set of BinPackingScheme's, each BinPackingScheme is a 2-D array -
    # If Pij = 0, it means block i is not in page j;
    # If Pij = 1, it means block i is in page j.
    return P


def test_pack():
    # Unique items in tensor
    I = [("t1", 0, 1), ("t1", 0, 0), ("t1", 1, 1)]
    P = pack(I, 2)
    import pdb
    pdb.set_trace()


"""
1: INPUT1: P_star (a previous list of optimal bin-packing schemes - a set of bin sets)
2: INPUT2: 𝑡 (a new tensor to be added - a set of tensor blocks)
3: INPUT3: 𝑙 (the maximum number of items for each bin)
4: OUTPUT: P = {𝑃𝑘 } (a list of optimal bin-packing schemes - a set of bin sets)
           A bin-packing scheme is a set of bins
"""
def adjust(P_star, t, l):
    # Again, t can have duplicates?
    I = set(t)

    minNumBins = math.inf
    P = set()

    # P_k is a BinPackingScheme which is a 2-D array -
    # If Pij = 0, it means block i is not in page j;
    # If Pij = 1, it means block i is in page j.
    for P_k in P_star:
        bin_set = P_k.findMinBinsMaxCover(I)
        I_delta = I - set().union(*bin_set)
        I_delta = list(I_delta)

        if not I_delta:
            if P_k.numBins == minNumBins:
                P = P.union(set([P_k]))
            else:
                if P_k.numBins < minNumBins:
                    P = set([P_k])
                    minNumBins = P_k.numBins
        else:
            P_prime = pack(I_delta, l)
            for P_dash in P_prime:
                P_new = P_dash.merge(P_k)
                if P_new.numBins == minNumBins:
                    P = P.union(set([P_new]))
                else:
                    if P_new.numBins < minNumBins:
                        P = set([P_new])
                        minNumBins = P_new.numBins

    return P
    

def test_adjust():
    # Can have repeated elements
    t1 = [("t1", 0, 1), ("t1", 0, 0), ("t1", 0, 0), ("t1", 1, 1)]
    t2 = [("t1", 0, 1), ("t1", 0, 0), ("t2", 1, 1)]
    l = 2
    P_star = pack(t1, l)

    P = adjust(P_star, t2, l)

    import pdb
    pdb.set_trace()


"""
INPUT1: T (A set of tensors, each tensor is a set of block ids) - each can have duplicates?
INPUT2: l (the maximum number of items that can be held in a bin)
OUTPUT: P={𝑃𝑖} (a list of optimal bin-packing schemes)
"""
def bin_pack_dp(T, l):
    initialized = False
    P_star = None
    for t_i in T:
        if not initialized:
            P_star = pack(t_i, l)
            initialized = True
        else:
            P_star = adjust(P_star, t_i, l)

    return P_star


def order_tensors_by_size(T):
    return sorted(T, key=lambda x: len(x), reverse=True)

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

    ordered_items = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
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
    
    tensors = order_tensors_by_size(T)
    # Add tensor t1
    items = order_tensor_blocks_by_freq(T, tensors[0])

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
        bin_set = p_i_j.findMinBinsMaxCover(tensors[i - 1])
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
                p_i_j.mark(s, u)

            numBins = numBins + math.ceil(len(remaining_items) / l)

    return set([p_i_j])


if __name__ == "__main__":
    test_pack()
    # test_adjust()