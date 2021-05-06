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
from bulk_labelling.embedding import get_embeddingset, get_language_array



@streamlit.cache
def prepare_data(lang, transformer, textlist=None):
    

    encoding, texts = get_language_array(lang, textlist)
    embset = get_embeddingset(encoding, texts)
    result = embset.transform(transformer)
    
    return result

@streamlit.cache
def make_plot(lang, transformer, textlist=None):
    return prepare_data(lang, transformer, textlist).plot_interactive(annot=False).properties(width=1000, height=500, title=type(lang).__name__)



