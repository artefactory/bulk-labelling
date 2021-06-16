import json
from lib.utils.load_config import load_languages, load_transformer
from lib.utils.embedding import prepare_data
from lib.custom_whatlies import EmbeddingSet, embedding
from lib.utils.processing import remove_stopwords_from_column
import time
import logging
import numpy as np
import pandas as pd
import streamlit
from lib.utils.embedding import get_embeddingset
import pathlib
import os


def compute_to_cache(
    embedding_language,
    languages_dict,
    transformer_option,
    transformers_dict,
    dataset,
    option,
    column_name,
    my_bar,
    sample_data,
    remove_stopwords,
    dataset_language,
    custom_stopwords,
):
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
    with streamlit.spinner(":hourglass: Loading language model..."):
        lang = load_languages(embedding_language, languages_dict)
    s1 = time.time()

    transformer = load_transformer(transformer_option, transformers_dict)
    s2 = time.time()

    if remove_stopwords:
        with streamlit.spinner(":hourglass: Computing embeddings..."):
            dataset_to_filter = dataset.copy()
            dataset_to_filter[column_name] = remove_stopwords_from_column(
                dataset_to_filter[column_name], dataset_language, custom_stopwords
            )
            temp_datasets = np.array_split(dataset_to_filter, 100)
    else:
        temp_datasets = np.array_split(dataset, 100)

    with streamlit.spinner(":hourglass: Computing embeddings..."):

        temp_embsets = []
        for k in range(len(temp_datasets)):
            textlist = (
                temp_datasets[k][column_name]
                .apply(lambda x: "None" if x == "" else x)
                .astype(str)
                .tolist()
            )
            temp_embset = prepare_data(lang, textlist)
            my_bar.progress(k / len(temp_datasets))
            # temp_embsets.append(temp_embset.to_dataframe().reset_index())
            temp_embsets.append(temp_embset)

    # embedding_df=pd.concat(temp_embsets)

    embarray_texts = []
    for k in temp_embsets:
        embarray_texts += k.to_names_X()[0]
    embarray_encoding = np.vstack([k.to_names_X()[1] for k in temp_embsets])

    with streamlit.spinner(":hourglass: Reducing embedding dimensions..."):

        embedding_df = (
            get_embeddingset(embarray_encoding, embarray_texts)
            .transform(transformer)
            .to_dataframe()
            .reset_index()
        )

    s3 = time.time()

    embedding_df["labelling_uuid"] = dataset.labelling_uuid
    embedding_df.to_csv("data/plotting_data/cache/cache.csv", index=False)

    json_cache = {
        "dataset": option,
        "column": column_name,
        "language_model": embedding_language,
        "reduction_algorithm": transformer_option,
        "sampled": sample_data,
        "remove_stopwords": remove_stopwords,
        "dataset_language": dataset_language,
        "custom_stopwords": custom_stopwords,
    }
    with open("data/plotting_data/cache/cache.json", "w", encoding="utf-8") as f:
        json.dump(json_cache, f, ensure_ascii=False, indent=4)
    logging.info(f"loading language : {s1-start}")
    logging.info(f"loading transformer : {s2-s1}")
    logging.info(f"preparing data total time : {s3-s2}")
    return embedding_df



def fetch_embedding_df_from_cache(
    progress_bar,
    embedding_language,
    languages_dict,
    transformer_option,
    transformers_dict,
    dataset,option,
    column_name,
    sample_data,
    remove_stopwords,
    dataset_language,
    custom_stopwords,
    compute):
    embedding_df=None
    if ("cache.json" not in os.listdir("data/plotting_data/cache")) and compute:
                my_bar = progress_bar.progress(0)

                embedding_df = compute_to_cache(
                    embedding_language,
                    languages_dict,
                    transformer_option,
                    transformers_dict,
                    dataset,
                    option,
                    column_name,
                    my_bar,
                    sample_data,
                    remove_stopwords,
                    dataset_language,
                    custom_stopwords,
                )
                progress_bar.success(":heavy_check_mark: Your data is ready!")

    if ("cache.json" in os.listdir("data/plotting_data/cache")) and compute:

        with open("data/plotting_data/cache/cache.json", encoding='utf-8') as f:
            cached_data = json.load(f)
        json_cache = {
            "dataset": option,
            "column": column_name,
            "language_model": embedding_language,
            "reduction_algorithm": transformer_option,
            "sampled": sample_data,
            "remove_stopwords": remove_stopwords,
            "dataset_language": dataset_language,
            "custom_stopwords": custom_stopwords,
        }
        if cached_data != json_cache:
            my_bar = progress_bar.progress(0)
            embedding_df = compute_to_cache(
                embedding_language,
                languages_dict,
                transformer_option,
                transformers_dict,
                dataset,
                option,
                column_name,
                my_bar,
                sample_data,
                remove_stopwords,
                dataset_language,
                custom_stopwords,
            )
            progress_bar.success(":heavy_check_mark: Your data is ready!")
        else:
            embedding_df = pd.read_csv("data/plotting_data/cache/cache.csv")
    
    else:
        try:
            embedding_df = pd.read_csv("data/plotting_data/cache/cache.csv")
        except:
            pass
    return embedding_df




def compute_stopwords():
    pass

def compute_embeddings(stopwords_df):
    pass

def compute_dimension_reduction(embeddings_df):
    pass



def read_cache(path):
    return pd.read_csv(path)

def write_cache(path,df):
    df.to_csv(path,index=False)

def clear_embeddings_cache():
    pass

def clear_reduction_cache():
    pass








def clear_cache():
    """clears the handmade cache"""
    [
        f.unlink()
        for f in pathlib.Path("data/plotting_data/cache").glob("*")
        if (f.is_file() and not os.path.basename(f).startswith(".git"))
    ]


def export_cache(dataset_name, path):
    """exports cache to wanted path once labelising is done

    Args:
        dataset_name (str): name of dataset to save to path
        path (str): path root to save dataset to
    """
    try:
        dataset = pd.read_csv("data/plotting_data/cache/cache.csv")
        dataset.to_csv(path + f"{dataset_name}.csv", index=False)
    except:
        pass
