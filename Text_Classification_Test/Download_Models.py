# general template to evaluate text classification model

# download text classification dataset
import gdown
import os

#if not os.path.isfile('model_deduplication_dataset.zip'):
#    model_deduplication_dataset_url = 'https://drive.google.com/uc?id=1nYzDDSJGkjCsVQI4gbSdC3DafQ7Ez107'
#    gdown.download(model_deduplication_dataset_url, output=None, quiet=False)
#    !unzip -qqq model_deduplication_dataset.zip

w2v_wiki500_yelp_embed_nontrainable_model_url = 'https://drive.google.com/uc?id=1-6T6c5MaaceARapMnPBEj0KP_P1-rg0y'
gdown.download(w2v_wiki500_yelp_embed_nontrainable_model_url, output=None, quiet=False)

w2v_wiki500_yelp_embed_trainable_model_url = 'https://drive.google.com/uc?id=1GgVaiexh643C7LlVH0qJHqlJxBZo71M0'
gdown.download(w2v_wiki500_yelp_embed_trainable_model_url, output=None, quiet=False)

w2v_wiki500_imdb_embed_nontrainable_model_url = 'https://drive.google.com/uc?id=1jG2UjS75KG4pOeRKCWYdZd1o3TQlewfy'
gdown.download(w2v_wiki500_imdb_embed_nontrainable_model_url, output=None, quiet=False)

w2v_wiki500_imdb_embed_trainable_model_url = 'https://drive.google.com/uc?id=1-1I6r7kBQhwyHi5_Xatrwz_VFVM3H8GH'
gdown.download(w2v_wiki500_imdb_embed_trainable_model_url, output=None, quiet=False)

w2v_wiki500_civil_comment_embed_trainable_model_url = 'https://drive.google.com/uc?id=1--bCnYYoe0mXseM783qqkncVMQVQmbhy'
gdown.download(w2v_wiki500_civil_comment_embed_trainable_model_url, output=None, quiet=False)