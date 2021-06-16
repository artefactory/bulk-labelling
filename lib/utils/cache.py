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

    if sample_data:
        sampled_data = dataset.sample(1000)
    else:
        sampled_data = dataset.copy()

    if remove_stopwords:
        with streamlit.spinner(":hourglass: Computing embeddings..."):
            dataset_to_filter = sampled_data.copy()
            dataset_to_filter[column_name] = remove_stopwords_from_column(
                dataset_to_filter[column_name], dataset_language, custom_stopwords
            )
            temp_datasets = np.array_split(dataset_to_filter, 100)
    else:
        temp_datasets = np.array_split(sampled_data, 100)

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
    dataset,
    option,
    column_name,
    sample_data,
    remove_stopwords,
    dataset_language,
    custom_stopwords,
    compute,
):
    embedding_df = None
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

        with open("data/plotting_data/cache/cache.json", encoding="utf-8") as f:
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


def compute_stopwords(
    remove_stopwords, sampled_data, column_name, dataset_language, custom_stopwords
):
    if remove_stopwords:
        dataset_to_filter = sampled_data.copy()
        dataset_to_filter[column_name] = remove_stopwords_from_column(
            dataset_to_filter[column_name], dataset_language, custom_stopwords
        )
        return dataset_to_filter
    else:
        return sampled_data


def compute_samples(sample_data, df):

    if sample_data:
        return df.sample(1000)
    else:
        return df


def compute_embeddings(
    df, embedding_language, languages_dict, progress_bar, column_name
):
    my_bar = progress_bar.progress(0)
    with streamlit.spinner(":hourglass: Loading language model..."):
        lang = load_languages(embedding_language, languages_dict)

    temp_datasets = np.array_split(df, 100)
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
            temp_embsets.append(temp_embset)

    embarray_texts = []
    for k in temp_embsets:
        embarray_texts += k.to_names_X()[0]
    embarray_encoding = np.vstack([k.to_names_X()[1] for k in temp_embsets])
    df = (
        get_embeddingset(embarray_encoding, embarray_texts).to_dataframe().reset_index()
    )
    return df


def compute_dimension_reduction(df, transformer_option, transformers_dict):
    with streamlit.spinner(":hourglass: Reducing embedding dimensions..."):
        transformer = load_transformer(transformer_option, transformers_dict)
        embarray_encoding = df.drop(columns="index").values
        embarray_texts = df["index"]
        embedding_df = (
            get_embeddingset(embarray_encoding, embarray_texts)
            .transform(transformer)
            .to_dataframe()
            .reset_index()
        )
        return embedding_df


def read_cache(name):
    return pd.read_csv("data/plotting_data/cache/{}".format(name))


def write_cache(name, df):
    df.to_csv("data/plotting_data/cache/{}".format(name), index=False)


def compute_all(
    sample_data,
    df,
    remove_stopwords,
    column_name,
    embedding_language,
    languages_dict,
    progress_bar,
    transformer_option,
    transformers_dict,
    custom_stopwords,
    dataset_language
):
    df = compute_samples(sample_data, df)
    write_cache("sampled_cache.csv", df)
    df = compute_stopwords(remove_stopwords, df, column_name,dataset_language,custom_stopwords)
    write_cache("stopwords_cache.csv", df)
    df = compute_embeddings(
        df, embedding_language, languages_dict, progress_bar, column_name
    )
    write_cache("embedding_cache.csv", df)
    df = compute_dimension_reduction(df, transformer_option, transformers_dict)
    return df


def compute_cache_test(
    progress_bar,
    embedding_language,
    languages_dict,
    transformer_option,
    transformers_dict,
    dataset,
    option,
    column_name,
    sample_data,
    remove_stopwords,
    dataset_language,
    custom_stopwords,
    compute,
):
    no_cache = False

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


    if ("cache.json" not in os.listdir("data/plotting_data/cache")) and compute:
        streamlit.write("no cache. computing...")
        my_bar = progress_bar.progress(0)

        df = compute_all(
            sample_data,
            dataset,
            remove_stopwords,
            column_name,
            embedding_language,
            languages_dict,
            progress_bar,
            transformer_option,
            transformers_dict,
            dataset_language,
            custom_stopwords
        )

        no_cache = True
        progress_bar.success(":heavy_check_mark: Your data is ready!")

        
        with open("data/plotting_data/cache/cache.json", "w", encoding="utf-8") as f:
            json.dump(json_cache, f, ensure_ascii=False, indent=4)

        return df
        

    if (
        "cache.json" in os.listdir("data/plotting_data/cache")
        and compute
        and not no_cache
    ):

    
        with open("data/plotting_data/cache/cache.json", encoding="utf-8") as f:
            cached_data = json.load(f)

        # CACHED DATASET
        if (
            cached_data["dataset"] != json_cache["dataset"]
            or cached_data["column"] != json_cache["column"]
        ):
            df = dataset.copy()
            write_cache("dataset_cache.csv", df)
            dataset_changed = True

        else:
            try:
                df = read_cache("dataset_cache.csv")
                dataset_changed = False
            except:
                df = dataset.copy()
                write_cache("dataset_cache.csv", df)
                dataset_changed = False

        # CACHED SAMPLED DATASET
        if cached_data["sampled"] != json_cache["sampled"] or dataset_changed:
            df = compute_samples(sample_data, df)
            write_cache("sampled_cache.csv", df)
            sampled_changed = True
        else:
            try:
                df = read_cache("sampled_cache.csv")
                sampled_changed = False
            except:
                df = compute_samples(sample_data, df)
                write_cache("sampled_cache.csv", df)

                sampled_changed = True

        # CACHED STOPWORDS
        if (
            cached_data["custom_stopwords"] != json_cache["custom_stopwords"]
            or cached_data["remove_stopwords"] != json_cache["remove_stopwords"]
            or cached_data["dataset_language"] != json_cache["dataset_language"]
            or dataset_changed
            or sampled_changed
        ):
            df = compute_stopwords(
                remove_stopwords, df, column_name, dataset_language, custom_stopwords
            )
            write_cache("stopwords_cache.csv", df)
            stopwords_changed = True
        else:
            try:
                df = read_cache("stopwords_cache.csv")
                stopwords_changed = False
            except:
                df = compute_stopwords(
                    remove_stopwords,
                    df,
                    column_name,
                    dataset_language,
                    custom_stopwords,
                )
                write_cache("stopwords_cache.csv", df)
                stopwords_changed = True

        # CACHED LANGUAGE MODEL
        if (
            cached_data["language_model"] != json_cache["language_model"]
            or sampled_changed
            or dataset_changed
            or stopwords_changed
        ):
            df = compute_embeddings(
                df, embedding_language, languages_dict, progress_bar, column_name
            )
            write_cache("embedding_cache.csv", df)
            model_changed = True
        else:
            try:
                df = read_cache("embedding_cache.csv")
                model_changed = False
            except:
                df = compute_embeddings(
                    df, embedding_language, languages_dict, progress_bar, column_name
                )
                write_cache("embedding_cache.csv", df)
                model_changed = True

        if (
            cached_data["reduction_algorithm"] != json_cache["reduction_algorithm"]
            or sampled_changed
            or dataset_changed
            or model_changed
            or stopwords_changed
        ):

            dr_changed = True
            df = compute_dimension_reduction(df, transformer_option, transformers_dict)
            write_cache("cache.csv", df)
        else:
            try:
                df = read_cache("cache.csv")

            except:
                df = compute_dimension_reduction(
                    df, transformer_option, transformers_dict
                )
                write_cache("cache.csv", df)

        with open("data/plotting_data/cache/cache.json", "w", encoding="utf-8") as f:
            json.dump(json_cache, f, ensure_ascii=False, indent=4)

        progress_bar.success(":heavy_check_mark: Your data is ready!")
        return df

    else:
        try:
            df=read_cache('cache.csv')
            return df
        except:
            return None

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
