
import streamlit
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datasets
import pathlib
import yaml
from streamlit.uploaded_file_manager import UploadedFile
import tensorflow as tf
import tensorflow_hub
from whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage, SpacyLanguage
from whatlies.language import TFHubLanguage
from whatlies import Embedding, EmbeddingSet, embedding
from whatlies.transformers import Pca, Umap, Tsne, Lda
from sentence_transformers import SentenceTransformer
from preshed.maps import PreshMap
from cymem.cymem import Pool
import json
import uuid
from bulk_labelling.embedding import get_embeddingset, get_language_array


@streamlit.cache
def load_dataset(dataset_name,datasets_dict):
    if dataset_name in datasets_dict:
        # if dataset_name != '-':
        #     if dataset_name == 'bing_coronavirus_query_set':
        #         dataset = datasets.load_dataset("bing_coronavirus_query_set", queries_by="country", start_date="2020-09-01", end_date="2020-09-30")
        #     else:
        #         dataset = datasets.load_dataset(dataset_name)
        #     dataset = pd.DataFrame.from_dict(dataset['train'])
        #     return dataset
        pass
    else:
        dataset = pd.read_csv(f'data/datasets/{dataset_name}')
        if 'labelling_uuid' not in dataset.columns:
            dataset['labelling_uuid'] = [uuid.uuid4() for _ in range(len(dataset.index))]
        dataset.to_csv(f'data/datasets/{dataset_name}',index=False)
        return dataset

def load_languages(language, languages_dict):
    return eval(languages_dict[language])

def load_transformer(option,transformers_dict):
    transformer = eval(transformers_dict[option])
    return transformer

def load_config():
    result=yaml.load(open('config/config.yml'))
    embedding_framework=pd.DataFrame.from_dict(result['embedding_framework'])

    return result['languages_dict'],result['transformers_dict'],result['datasets_dict'],embedding_framework


