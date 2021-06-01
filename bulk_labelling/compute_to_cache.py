import json
from bulk_labelling.load_config import load_languages, load_transformer
from bulk_labelling.plotting import prepare_data


def compute_to_cache(embedding_language, languages_dict, transformer_option, transformers_dict, dataset, option, column_name):
    lang = load_languages(embedding_language, languages_dict)
    transformer = load_transformer(
        transformer_option, transformers_dict)
    textlist = dataset[column_name].astype(str).tolist()
    embset = prepare_data(lang, transformer, textlist)
    embedding_df = embset.to_dataframe().reset_index()
    embedding_df['labelling_uuid'] = dataset.labelling_uuid
    embedding_df.to_csv(
        'data/plotting_data/cache/cache.csv', index=False)
    json_cache = {'dataset': option, 'column': column_name,
                  'language_model': embedding_language, 'reduction_algorithm': transformer_option}
    with open('data/plotting_data/cache/cache.json', 'w', encoding='utf-8') as f:
        json.dump(json_cache, f, ensure_ascii=False, indent=4)
    return embedding_df
