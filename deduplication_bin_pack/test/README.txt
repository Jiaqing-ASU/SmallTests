For the input of these 3 Python scripts is a dictionary object saved in a .npy file which has the following format:

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
If you would like to run the most greedy algorithm, please run greedy_test.py.
if you would like to run the pruning greedy algorithm based on dp algorithm, please run dp_greedy_test.py.
If you would like to run the real dynamic programming algorithm, please run dp_test.py

When you run these 3 scripts, you could also change the number of blocks each page can mostly contain by change the following value:
#blocks_in_page = 10

After you feel good with all the above settings, you could run the python script you need and then it will output the time of the algorithm running and also the numbers of pages need from the terminal. 

If you run the greedy_test.py, you will also see the output file under the same folder. It is also a dictionary object saved in a .npy file named greedy_page_pack_output.npy. This file has the following format:

Key: Value


block_page_mapping:      a list of block IDs mapped to page IDs
						 e.g. 0:211 means block #0 is in page #211
tensor_page_mapping:	 a list of tensor IDs mapped to page IDs
						 e.g. 0: {185,183,90,27} means tensor #0 needs page #185, #183, #90 and #27

If you run the other 2 scripts, you will also have an output file under the same folder. It is a dictionary object saved in a .npy file named dp_page_pack_output.npy or dp_greedy_page_pack_output.npy. However, since these 2 algorithms might have more than 1 optimal plan, the output might be multiple:


Key: Value


block_page_mapping0:     a list of block IDs mapped to page IDs
                         e.g. 0:211 means block #0 is in page #211
tensor_page_mapping0:    a list of tensor IDs mapped to page IDs
                         e.g. 0: {185,183,90,27} means tensor #0 needs page #185, #183, #90 and #27

block_page_mapping1:     a list of block IDs mapped to page IDs
                         e.g. 0:211 means block #0 is in page #211
tensor_page_mapping1:    a list of tensor IDs mapped to page IDs
                         e.g. 0: {185,183,90,27} means tensor #0 needs page #185, #183, #90 and #27

............