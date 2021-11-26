# Introduction

This repository contains all the code for text classification and extreme classification testing on Tensorflow.

# Environment
AWS r4.xlarge for CPU testing and AWS g4dn.2xlarge for GPU testing.
AMI Deep Learning Base AMI (Ubuntu 18.04) Version 45.0.
TensorFlow 2.6.2

## Install TensorFlow Hub
```
$pip3 install --upgrade tensorflow-hub
```

## Install Postgres
```
$sudo apt update
$sudo apt install postgresql postgresql-contrib
```

# Text Classification
For text classification, please go to Text_Classification_Test folder.

## Download the models
Please download the model when running for the first time. The model is saved locally in H5 format. Please run:
```
$python3 Download_Models.py
```

## Change the weights(Optional)
At this time, the weights of the model downloaded to the local is float32. Our tests for text classification based on look up function are using float32 precision and the tests based on matrix multiplication are using double precision. We will load one of the previous models and then change the wights from float32 to double while testing in matrix multiplication. However, if you would like to save the weights of the models as double precision, please run Change_Weights_Float_to_Double.py: (This is optional. Not do so will not affect the testing results.)
```
$python3 Change_Weights_Float_to_Double.py
```

## Generate the inputs and save the inputs to CSV and Postgres
After processing the model, we need to process the input. Input needs to be saved as CSV or imported into the Postgres database. Please first make sure that there is one database with user names and passwords are both "postgres" in your running environment. We will create tables inside it. You could use the following command to meet the above requirements.
```
$sudo -i -u postgres
$psql
postgres=# \password postgres
Enter new password: <new-password>
postgres=# \q
```

And then run the following three python scripts in turn.
$python3 Load_Data_to_Postgres.py
$python3 Load_Postgres_M_Col_Text.py
$python3 Save_Input_to_CSV.py

## Run the test program
There are 2 kinds of methods are testing for text classification. Our tests for text classification based on the look up function are using float32 precision and the tests based on matrix multiplication are using double precision. word2vec-inference-MM-exp.py is using matrix multiplication and word2vec-inference-exp.py is using the look up function.
```
$python3 word2vec-inference-MM-exp.py
$python3 word2vec-inference-exp.py
```

# Extreme Classification
For extreme classification, please go to Extreme_Classification_Test folder.

## Create the model
Please create the model when running for the first time. The model is saved locally in H5 format. Please run:
```
$python3 Build_Model.py
```

You could save the weights of the model either in float32 or double. Our tests are under double precision, so the default setting is double precision.

## Generate the inputs and save the inputs to CSV and Postgres
After processing the model, we need to process the input. Input needs to be saved as CSV or imported into the Postgres database. Please first make sure that there is one database with user names and passwords are both "postgres" in your running environment. We will create tables inside it. You could use the following command to meet the above requirements.
```
$sudo -i -u postgres
$psql
postgres=# \password postgres
Enter new password: <new-password>
postgres=# \q
```

And then run the following three python scripts in turn.
$python3 Load_Data_to_Postgres.py
$python3 Load_Postgres_M_Col.py
$python3 Save_Input_to_CSV.py

## Run the test program
The only one testing method for text classification, which is based on matrix multiplication is using double precision.
```
$python3 extreme_classification_model_exp.py
```