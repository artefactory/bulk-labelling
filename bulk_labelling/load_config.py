
import streamlit
import pandas as pd
import datasets
import yaml
from bulk_labelling.custom_whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage, SpacyLanguage
from bulk_labelling.custom_whatlies.language import TFHubLanguage
from bulk_labelling.custom_whatlies.embedding import Embedding
from bulk_labelling.custom_whatlies.embeddingset import EmbeddingSet
from bulk_labelling.custom_whatlies.transformers import Pca, Umap, Tsne
from sentence_transformers import SentenceTransformer
from preshed.maps import PreshMap
from cymem.cymem import Pool
import uuid
from bulk_labelling.embedding import get_embeddingset, get_language_array
import os


@streamlit.cache
def load_dataset(dataset_name, datasets_dict):
    if dataset_name in datasets_dict:
        if dataset_name != '-':
            if dataset_name == 'bing_coronavirus_query_set':
                dataset = datasets.load_dataset(
                    "bing_coronavirus_query_set", queries_by="country", start_date="2020-09-01", end_date="2020-09-30")
            else:
                dataset = datasets.load_dataset(dataset_name)
            dataset = pd.DataFrame.from_dict(dataset['train'])
            if 'labelling_uuid' not in dataset.columns:
                dataset['labelling_uuid'] = [uuid.uuid4()
                                             for _ in range(len(dataset.index))]
            dataset.to_csv(f'data/datasets/{dataset_name}.csv', index=False)
            return dataset
        pass
    else:
        dataset = pd.read_csv(f'data/datasets/{dataset_name}')
        if 'labelling_uuid' not in dataset.columns:
            dataset['labelling_uuid'] = [uuid.uuid4()
                                         for _ in range(len(dataset.index))]
        dataset.to_csv(f'data/datasets/{dataset_name}', index=False)
        return dataset


def load_languages(language, languages_dict):
    return eval(languages_dict[language])


def load_transformer(option, transformers_dict):
    transformer = eval(transformers_dict[option])
    return transformer


def load_config():
    result = yaml.load(open('config/config.yml'))
    embedding_framework = pd.DataFrame.from_dict(result['embedding_framework'])

    return result['languages_dict'], result['transformers_dict'], result['datasets_dict'], embedding_framework
