For the input of this Python script is a dictionary object saved in a .npy file which has the following format:

Key: Value

list_blocks:              a list that stores the distinct blocks, the size is 
                          equals to num_distinct_blocks
tensor_mapping:           a list of tensor mapping whose length is the number  
                          of tensor. Each element is a dictionary that stores
                          the mapping between the block index and its distinct 
                          block ID. e.g tensor_mapping[0][(1,1)] = 50 means the 
                          block (1,1) in the 0th tensor is mapped to the distinct
                          block 50.
blocked_tensor_dimension: a list of dimensions for each blocked tensor
block_size:               the size of the block

And you could directly change the line:
#input = np.load('detector_output.npy', allow_pickle=True).item()
Replace the 'detector_output.npy' as the input file you would like to test and also you could use the sys to set the input file as a parameter by using the following lines:
#file_path = sys.argv[1]
#input = np.load(file_path, allow_pickle=True).item()

Besides the input of this script, there are also 3 algothrims you could run.
Find the following lines in the test.py:

#P, tensor_page_mapping = bin_pack_greedy(list_of_tensors, blocks_in_page)
#P = bin_pack_dp_greedy(list_of_tensors, blocks_in_page)
#P = bin_pack_dp(list_of_tensors, blocks_in_page)

if you run the bin_pack_greedy, just delete the '#' before line:
#P, tensor_page_mapping = bin_pack_greedy(list_of_tensors, blocks_in_page)

And if you run the other 2 algorithms, please delete the '#' before them and also commit the following line:
#output['tensor_page_mapping'] = tensor_page_mapping

The reason is after fix the bugs, the previous parts of code to generate the tensor_page_mapping does not work. And I need to generate a new method to get the tensor_page_mapping.[TODO]

When you run this script, you could also change the number of blocks each page can mostly contain by change the following value:
#blocks_in_page = 10

After you feel good with all the above settings, you could run the test.py and then it will output the time of the algorithm running and also the numbers of pages need from the terminal. 

If you run the most greedy algorithm, you will also see the output file under the same folder. It is also a dictionary object saved in a .npy file named page_pack_output.npy. This file has the following format:

Key: Value


block_page_mapping:      a list of block IDs mapped to page IDs
						 e.g. 0:211 means block #0 is in page #211
tensor_page_mapping:	 a list of tensor IDs mapped to page IDs
						 e.g. 0: {185,183,90,27} means tensor #0 needs page #185, #183, #90 and #27

If you run the other 2 algorithms, you will also have an output file under the same folder. It is a dictionary object saved in a .npy file named page_pack_output.npy. However, this file will only have the following information:

Key: Value

block_page_mapping:      a list of block IDs mapped to page IDs
						 e.g. 0:211 means block #0 is in page #211