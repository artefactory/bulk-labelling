
import streamlit
import os
import pandas as pd
from bulk_labelling.custom_whatlies.language import CountVectorLanguage, UniversalSentenceLanguage, BytePairLanguage, SentenceTFMLanguage, SpacyLanguage
from bulk_labelling.custom_whatlies.language import TFHubLanguage
from bulk_labelling.custom_whatlies.embedding import Embedding
from bulk_labelling.custom_whatlies.embeddingset import EmbeddingSet
from bulk_labelling.custom_whatlies.transformers import Pca, Umap, Tsne
from preshed.maps import PreshMap
from cymem.cymem import Pool
import json
from streamlit_bokeh_events import streamlit_bokeh_events
import matplotlib.pyplot as plt

from bulk_labelling.load_config import load_languages, load_transformer, load_dataset, load_config
from bulk_labelling.embedding import get_embedding, get_embeddingset, get_language_array
from bulk_labelling.compute_to_cache import compute_to_cache
from bulk_labelling.plotting import prepare_data, make_plot, make_interactive_plot, suggest_clusters, clear_cache,generate_wordcloud
from bulk_labelling.pages import plain_plot, cluster_suggestion


def write():

    languages_dict, transformers_dict, datasets_dict, Embedding_frameworks_dataframe = load_config()

    streamlit.sidebar.title('Bulk labelling')
    dataset_upload = streamlit.sidebar.beta_expander('1. Select your dataset')

    option_box = dataset_upload.empty()

    dataset = None
    dataframe_preview = dataset_upload.empty()
    uploaded_file = dataset_upload.file_uploader("Add a custom dataset")
    uploaded_file_name = dataset_upload.text_input(
        "Custom dataset name", value='')

    if (uploaded_file is not None):
        uploaded_dataset = pd.read_csv(uploaded_file)
        dataset = uploaded_dataset.copy()
        if uploaded_file_name != '':
            uploaded_dataset.to_csv(
                'data/datasets/{}.csv'.format(uploaded_file_name), index=False)

    available_datasets = datasets_dict + \
        [i for i in os.listdir('data/datasets') if '.csv' in i]
    option = option_box.selectbox('Dataset:', available_datasets, index=0)
    dataset = load_dataset(option, datasets_dict)

    try:
        dataframe_preview.dataframe(dataset.head())
    except Exception:
        pass

    
    column_select = streamlit.sidebar.beta_expander(
        '2. Select column for analysis')

    embedding = streamlit.sidebar.beta_expander(
        "3. Select your embedding framework")
    embedding_lang_select = embedding.beta_container()
    embedding_lang = embedding.beta_container()
    languages_embedding = embedding_lang_select.multiselect('Embedding framework languages', [
                                                            'english', 'french', 'multilingual'], ['english', 'french', 'multilingual'])
    embedding_language = embedding_lang.selectbox(
        'Embedding framework', Embedding_frameworks_dataframe[Embedding_frameworks_dataframe.language.isin(languages_embedding)].framework.tolist())

    map = streamlit.sidebar.beta_expander("4. Select your DR algorithm")
    transformer_option = map.selectbox(
        'Dimension reduction framework', ('TSNE', 'PCA', 'Umap'))

    big_container=streamlit.beta_container()
    wordcloud_container = streamlit.beta_container()
    chart_container, options_container = big_container.beta_columns(2)
    info_container = options_container.empty()
    dataview_container = options_container.empty()
    name_select = options_container.empty()
    click_clear_container = options_container.empty()

    plot = None
    column_name = '-'
    embedding_df = None
    selected_data = pd.DataFrame()


    cluster_suggestion_sidebar = streamlit.sidebar.beta_expander(
        '5. Cluster suggestion')

    suggest_clusters = cluster_suggestion_sidebar.checkbox('Suggest clusters?')

    graph_type = 'Interactive labeling'
    if suggest_clusters:
        graph_type = 'cluster suggestion'

    space = streamlit.sidebar.text('')
    compute = streamlit.sidebar.button('compute embeddings')
    # validate = streamlit.sidebar.checkbox('refresh plot')
    if graph_type == 'Interactive labeling':
        show_labeled = streamlit.sidebar.checkbox('show labeled data')

    clear_cache_button = streamlit.sidebar.button('clear cache')
    clear_labels_button = streamlit.sidebar.button('clear labels')

    if clear_cache_button:
        clear_cache()

    if dataset is not None:
        column_name = column_select.selectbox(
            'columns', options=['-'] + dataset.columns.tolist())

    if column_name != '-':

        if ('cache.json' not in os.listdir('data/plotting_data/cache')) and compute:
            embedding_df = compute_to_cache(
                embedding_language, languages_dict, transformer_option, transformers_dict, dataset, option, column_name)

        if ('cache.json' in os.listdir('data/plotting_data/cache')) and compute:

            f = open('data/plotting_data/cache/cache.json')
            cached_data = json.load(f)
            json_cache = {'dataset': option, 'column': column_name,
                          'language_model': embedding_language, 'reduction_algorithm': transformer_option}
            if cached_data != json_cache:
                embedding_df = compute_to_cache(
                    embedding_language, languages_dict, transformer_option, transformers_dict, dataset, option, column_name)
            else:
                embedding_df = pd.read_csv(
                    'data/plotting_data/cache/cache.csv')




        else:
            try:
                embedding_df = pd.read_csv(
                    'data/plotting_data/cache/cache.csv')
            except:
                pass
        
        if embedding_df is not None:


            embedding_df = embedding_df.rename(columns={
                                                embedding_df.columns[0]: 'text', embedding_df.columns[1]: 'd1', embedding_df.columns[2]: 'd2'})
            
            if 'labels' not in embedding_df.columns or clear_labels_button:
                embedding_df['labels'] = 'None'
                embedding_df.to_csv(
                    'data/plotting_data/cache/cache.csv', index=False)
    try:
        if column_name != '-':
            if embedding_df is not None:
                try:
                    if graph_type == 'Interactive labeling':

                        plot = make_interactive_plot(embedding_df, show_labeled)

                        with chart_container:
                            result_lasso = streamlit_bokeh_events(
                                bokeh_plot=plot,
                                events="LASSO_SELECT",
                                key="hello",
                                refresh_on_update=True,
                                debounce_time=1)
                            if result_lasso:
                                if result_lasso.get("LASSO_SELECT"):
                                    selected_data = embedding_df.iloc[result_lasso.get("LASSO_SELECT")[
                                        "data"]]
                                    


                                    info_container.info(
                                        f'selected {len(selected_data)} rows')
                                    name_select_value = name_select.text_input(
                                        'Input selected data label')
                                    click_clear = click_clear_container.button(
                                        'clear label')
                                    if click_clear:
                                        name_select_value = name_select.text_input(
                                            'Input selected data label', value='', key=1)
                                    if name_select_value:
                                        selected_data['labels'] = name_select_value
                                        embedding_df.iloc[result_lasso.get("LASSO_SELECT")[
                                            "data"]] = selected_data
                                        embedding_df.to_csv(
                                            'data/plotting_data/cache/cache.csv', index=False)


                                    wordcloud=generate_wordcloud(selected_data.text.tolist())
                                    fig, ax = plt.subplots()
                                    wordcloud_fig=ax.imshow(wordcloud,interpolation="bilinear")
                                    ax.axis('off')
                                    plt.savefig('data/plotting_data/cache/wordcloud.png')
                                    dataview_container.dataframe(selected_data.head(30),height=150)
                                    # wordcloud_container.image('data/plotting_data/cache/wordcloud.png',use_column_width=True)




                except Exception as error:
                    streamlit.write(error)

                if graph_type == 'cluster suggestion':
                    cluster_suggestion.write(embedding_df)
    except:
        pass
