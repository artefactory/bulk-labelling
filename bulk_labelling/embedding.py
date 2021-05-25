
import streamlit
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datasets
import pathlib
from streamlit.uploaded_file_manager import UploadedFile
import tensorflow as tf
import tensorflow_hub
from whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage, SpacyLanguage
from whatlies.language import TFHubLanguage
from whatlies import Embedding, EmbeddingSet
from whatlies.transformers import Pca, Umap, Tsne, Lda
from sentence_transformers import SentenceTransformer
from preshed.maps import PreshMap
from cymem.cymem import Pool
import json


@streamlit.cache
def get_embedding(vec, text):
    return Embedding(text, vec)


@streamlit.cache
def get_embeddingset(veclist, textlist):
    return EmbeddingSet(*[get_embedding(veclist[q], textlist[q]) for q in range(len(textlist))])

@streamlit.cache(allow_output_mutation=True)
def get_language_array(lang, textlist=None,uuid=None):
    if isinstance(lang, EmbeddingSet):
        return lang.to_names_X()[1], lang.to_names_X()[0]
    if isinstance(lang, SentenceTransformer):
        encoding = lang.encode(textlist)
        return encoding, textlist
    else:
        return lang[textlist].to_names_X()[1], lang[textlist].to_names_X()[0]

def cluster(algo,embset):
    transformed=embset[['d1','d2']].values
    
    clustering = algo.fit(transformed)

    labels = clustering.labels_
    embed=embset.copy()
    embed['labels'] = labels
    embed['labels']=embed['labels'].astype(str)
    return embed
