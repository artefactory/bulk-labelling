import json
from lib.load_config import load_languages, load_transformer
from lib.plotting import prepare_data
from lib.custom_whatlies import EmbeddingSet
import time
import logging
import numpy as np
import pandas as pd
import streamlit
from lib.embedding import get_embeddingset


def compute_to_cache(embedding_language, languages_dict, transformer_option, transformers_dict, dataset, option, column_name, my_bar, sample_data):
    """wrapper function for the whole computation steps, including writing to cache.

    Args:
        embedding_language (str): user-chosen language model
        languages_dict (dict): dictionary of possible language models
        transformer_option (str): user-chosen transformer
        transformers_dict (dict): dictionary of possible dimension reduction models
        dataset (pd.dataFrame()): dataset with data to analyze
        option (str): dataset name
        column_name (str): column to analyze

    Returns:
        pd.DataFrame(): dataframe with encoded-transformed data
    """
    start = time.time()
    lang = load_languages(embedding_language, languages_dict)
    s1 = time.time()
    transformer = load_transformer(
        transformer_option, transformers_dict)
    s2 = time.time()
    temp_datasets = np.array_split(dataset, 100)

    # textlist=dataset[column_name].astype(str).tolist()
    # embedding_df=prepare_data(lang,transformer,textlist).to_dataframe().reset_index()

    temp_embsets = []
    for k in range(len(temp_datasets)):
        textlist = temp_datasets[k][column_name].astype(str).tolist()
        temp_embset = prepare_data(lang, transformer, textlist)
        my_bar.progress(k/len(temp_datasets))
        # temp_embsets.append(temp_embset.to_dataframe().reset_index())
        temp_embsets.append(temp_embset)

    # embedding_df=pd.concat(temp_embsets)

    embarray_texts = []
    for k in temp_embsets:
        embarray_texts += k.to_names_X()[0]
    embarray_encoding = np.vstack([k.to_names_X()[1] for k in temp_embsets])

    embedding_df = get_embeddingset(embarray_encoding, embarray_texts).transform(
        transformer).to_dataframe().reset_index()

    s3 = time.time()

    # embedding_df = embset.to_dataframe().reset_index()
    embedding_df['labelling_uuid'] = dataset.labelling_uuid
    embedding_df.to_csv(
        'data/plotting_data/cache/cache.csv', index=False)

    json_cache = {'dataset': option, 'column': column_name,
                  'language_model': embedding_language, 'reduction_algorithm': transformer_option, 'sampled': sample_data}
    with open('data/plotting_data/cache/cache.json', 'w', encoding='utf-8') as f:
        json.dump(json_cache, f, ensure_ascii=False, indent=4)
    logging.info(f'loading language : {s1-start}')
    logging.info(f'loading transformer : {s2-s1}')
    logging.info(f'preparing data total time : {s3-s2}')
    return embedding_df
