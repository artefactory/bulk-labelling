import json
from bulk_labelling.load_config import load_languages, load_transformer
from bulk_labelling.plotting import prepare_data
import time
import logging


def compute_to_cache(embedding_language, languages_dict, transformer_option, transformers_dict, dataset, option, column_name):
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
    s1=time.time()
    transformer = load_transformer(
        transformer_option, transformers_dict)
    s2=time.time()
    textlist = dataset[column_name].astype(str).tolist()
    embset = prepare_data(lang, transformer, textlist)
    s3=time.time()
    embedding_df = embset.to_dataframe().reset_index()
    embedding_df['labelling_uuid'] = dataset.labelling_uuid
    embedding_df.to_csv(
        'data/plotting_data/cache/cache.csv', index=False)

    json_cache = {'dataset': option, 'column': column_name,
                  'language_model': embedding_language, 'reduction_algorithm': transformer_option}
    with open('data/plotting_data/cache/cache.json', 'w', encoding='utf-8') as f:
        json.dump(json_cache, f, ensure_ascii=False, indent=4)
    logging.info(f'loading language : {s1-start}')
    logging.info(f'loading transformer : {s2-s1}')
    logging.info(f'preparing data total time : {s3-s2}')
    return embedding_df
