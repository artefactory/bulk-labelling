
import streamlit
import pandas as pd
import datasets
import yaml
from lib.custom_whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage, SpacyLanguage
from lib.custom_whatlies.language import TFHubLanguage
from lib.custom_whatlies.embedding import Embedding
from lib.custom_whatlies.embeddingset import EmbeddingSet
from lib.custom_whatlies.transformers import Pca, Umap, Tsne
from sentence_transformers import SentenceTransformer
import uuid
import os


@streamlit.cache
def load_dataset_from_list(dataset_name):
    """loads a dataset from the samples provided

    Args:
        dataset_name (str): dataset name from the sample dataset directory

    Returns:
        pd.DataFrame(): dataframe containing the data from the loaded dataset
    """
    if dataset_name == '-':
        pass
    else:
        dataset = pd.read_csv(f'data/datasets/{dataset_name}')
        if 'labelling_uuid' not in dataset.columns:
            dataset['labelling_uuid'] = [uuid.uuid4()
                                         for _ in range(len(dataset.index))]
        dataset.to_csv(f'data/datasets/{dataset_name}', index=False)
        return dataset


def load_dataset(uploaded_file, uploaded_file_name, datasets_dict, option_box):
    dataset = None

    # from the point of user perspective, which is more intuitive? does the uploaded dataset take precedence or the selected one?
    if (uploaded_file is not None):

        uploaded_dataset = pd.read_csv(uploaded_file)
        dataset = uploaded_dataset.copy()
        if uploaded_file_name != '':
            uploaded_dataset.to_csv(
                'data/datasets/{}.csv'.format(uploaded_file_name), index=False)

    available_datasets = datasets_dict + \
        [i for i in os.listdir('data/datasets') if '.csv' in i]

    option = option_box.selectbox('Dataset:', available_datasets, index=0)

    if dataset is None:
        dataset = load_dataset_from_list(option)
    return dataset,option


def load_languages(language, languages_dict):
    """loads the language from the string-dict loaded from the config

    Args:
        language (str): selectbox choice for the language model from the user
        languages_dict (dict): dict containing to-evaluate strings of the language

    Returns:
        tfhub language or transformers language or whatlies language: language model to encode texts
    """
    return eval(languages_dict[language])


def load_transformer(option, transformers_dict):
    """loads a dimension-reduction transformer from the users' choice.

    Args:
        option (str): selected option in the selectbox from the user
        transformers_dict (dict): dictionary of to-evaluate strings representing transformers

    Returns:
        sklearn transformer: transformer to be used to dimension-reduce language model output.
    """
    transformer = eval(transformers_dict[option])
    return transformer


def load_config():
    """loads config from config.yaml file

    Returns:
        list: packed values for language model dict, transformer dict, datasets dict, and embedding framework dataframe
    """
    result = yaml.load(open('config/config.yml'))
    embedding_framework = pd.DataFrame.from_dict(result['embedding_framework'])

    return result['languages_dict'], result['transformers_dict'], result['datasets_dict'], embedding_framework
