This repository contains all the code for text classification and extreme classification testing on Tensorflow. In each folder, there are 8 python scripts.

For text classification, please go to Text_Classification_Test folder.

Please download the model when running for the first time. The model is saved locally in H5 format. Please run:
$python3 Download_Models.py

At this time, the weights of the model downloaded to the local is float32. If you want to save the weights of the model as double, please run Change_Weights_Float_to_Double.py:
$python3 Change_Weights_Float_to_Double.py

After processing the model, we need to process the input. Input needs to be saved as CSV or imported into the Postgres database. Please first make sure that there is one database with user names and passwords are both "postgres" in your running environment. We will create tables inside it. And then run the following three scripts in turn.
$python3 Load_Data_to_Postgres.py
$python3 Load_Postgres_M_Col_Text.py
$python3 Save_Input_to_CSV.py

Finally, run the test program:
$python3 word2vec-inference-MM-exp.py
$python3 word2vec-inference-exp.py
$python3 Load_From_Postgres_M_Col_Text.py

For extreme classification, please go to Extreme_Classification_Test folder.

Please create the model when running for the first time. The model is saved locally in H5 format. Please run:
$python3 Build_Model.py

You could save the weights of the model either in float32 or double. Our tests are under double format.

After processing the model, we need to process the input. Input needs to be saved as CSV or imported into the Postgres database. Please first make sure that there is one database with user names and passwords are both "postgres" in your running environment. We will create tables inside it. And then run the following three scripts in turn.
$python3 Load_Data_to_Postgres.py
$python3 Load_Postgres_M_Col.py
$python3 Save_Input_to_CSV.py

Finally, run the test program:
$python3 extreme_classification_model_exp.py
$python3 Load_From_Postgres_M_Col.py