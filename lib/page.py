
import streamlit
import os
import pandas as pd
import json
from streamlit_bokeh_events import streamlit_bokeh_events
import logging


from lib.load_config import load_dataset_from_list,load_dataset, load_config
from lib.compute_to_cache import compute_to_cache
from lib.plotting import make_interactive_plot, clear_cache, generate_wordcloud, replace_labels
from lib.pages import cluster_suggestion


def write():

    ###########################
    #     LOAD CONFIG         #
    ###########################

    languages_dict, transformers_dict, datasets_dict, Embedding_frameworks_dataframe = load_config()

    plot = None
    column_name = '-'
    embedding_df = None

    ##########################
    #      SIDEBAR           #
    ##########################

    streamlit.sidebar.title('Bulk labelling')

    ##########################
    #   UPLOADING DATASET    #
    ##########################

    dataset_upload = streamlit.sidebar.beta_expander('1. Select your dataset')

    option_box = dataset_upload.empty()
    dataframe_preview = dataset_upload.empty()
    uploaded_file = dataset_upload.file_uploader("Add a custom dataset")
    uploaded_file_name = dataset_upload.text_input(
        "Custom dataset name", value='')

    # move this to wrapper
    # dataset = None
    # if (uploaded_file is not None):
    #     uploaded_dataset = pd.read_csv(uploaded_file)
    #     dataset = uploaded_dataset.copy()
    #     if uploaded_file_name != '':
    #         uploaded_dataset.to_csv(
    #             'data/datasets/{}.csv'.format(uploaded_file_name), index=False)

    # available_datasets = datasets_dict + \
    #     [i for i in os.listdir('data/datasets') if '.csv' in i]
    # option = option_box.selectbox('Dataset:', available_datasets, index=0)
    # dataset = load_dataset_from_list(option)

    dataset,option=load_dataset(uploaded_file,uploaded_file_name,datasets_dict,option_box)

    try:
        dataframe_preview.dataframe(dataset.head())
    except Exception:
        pass

        ##########################
        #   COLUMN SELECTION     #
        ##########################

    column_select = streamlit.sidebar.beta_expander(
        '2. Select column for analysis')

    if dataset is not None:
        column_name = column_select.selectbox(
            'columns', options=['-'] + dataset.columns.tolist())
        sample_data = column_select.checkbox('Sample a smaller dataset')
        column_select.info(
            'One would sample the dataset to speed up calculations for a pre-labelling exploration phase')
        if sample_data:
            dataset = dataset.sample(n=1000)

        ##################################
        #   LANGUAGE MODEL SELECTION     #
        ##################################

    embedding = streamlit.sidebar.beta_expander(
        "3. Select your embedding framework")
    embedding_lang_select = embedding.beta_container()
    embedding_lang = embedding.beta_container()
    languages_embedding = embedding_lang_select.multiselect('Embedding framework languages', [
                                                            'english', 'french', 'multilingual'], ['english', 'french', 'multilingual'])
    embedding_language = embedding_lang.selectbox(
        'Embedding framework', Embedding_frameworks_dataframe[Embedding_frameworks_dataframe.language.isin(languages_embedding)].framework.tolist())

    #######################################
    #   DIMENSION REDUCTION SELECTION     #
    #######################################

    map = streamlit.sidebar.beta_expander("4. Dimension reduction algorithm")
    transformer_option = map.selectbox(
        'Dimension reduction framework', ('TSNE', 'PCA', 'Umap'))

    #######################################
    #           CLUSTER SUGGESTION        #
    #######################################

    cluster_suggestion_sidebar = streamlit.sidebar.beta_expander(
        '5. Cluster suggestion')

    suggest_clusters = cluster_suggestion_sidebar.checkbox('Suggest clusters?')
    suggest_clusters_slider = cluster_suggestion_sidebar.beta_container()
    if suggest_clusters:
        xi = suggest_clusters_slider.checkbox(
            'use xi method')
        epsilon = suggest_clusters_slider.slider(
            'epsilon', 0.0, 10.0, 4.0, 0.1)
        min_samples = suggest_clusters_slider.slider(
            'min_samples', 1, 100, 50, 1)
        min_cluster_size = suggest_clusters_slider.slider(
            'min_cluster_size', 1, 100, 30, 1)
        if xi:
            xi_value = suggest_clusters_slider.slider(
                'xi value', 0.0, 1.0, 0.05, 0.01)
        else:
            xi_value = 0.01

        ##############################
        #       VARIOUS BUTTONS      #
        ##############################

    space = streamlit.sidebar.text('')
    compute = streamlit.sidebar.button('compute embeddings')
    show_labeled = streamlit.sidebar.checkbox('show labeled data')
    clear_cache_button = streamlit.sidebar.button('clear cache')
    clear_labels_button = streamlit.sidebar.button('clear labels')

    if clear_cache_button:
        clear_cache()

    ##########################
    #   PAGE STRUCTURE       #
    ##########################

    big_container = streamlit.beta_container()
    progress_indicator = big_container.beta_container()
    progress_bar = progress_indicator.empty()
    wordcloud_container = streamlit.beta_container()
    chart_container_over, options_container = big_container.beta_columns(2)
    chart_container = chart_container_over.empty()
    info_container = options_container.empty()
    dataview_container = options_container.empty()
    name_select = options_container.empty()
    click_clear_container = options_container.empty()

    #####################################
    #       COMPUTING AND DISPLAY       #
    #####################################

    if column_name != '-':

        if ('cache.json' not in os.listdir('data/plotting_data/cache')) and compute:
            my_bar = progress_bar.progress(0)
            with streamlit.spinner('Computing embeddings...'):
                embedding_df = compute_to_cache(
                    embedding_language,
                    languages_dict,
                    transformer_option,
                    transformers_dict,
                    dataset,
                    option,
                    column_name,
                    my_bar,
                    sample_data)
            progress_bar.empty()

        if ('cache.json' in os.listdir('data/plotting_data/cache')) and compute:

            f = open('data/plotting_data/cache/cache.json')
            cached_data = json.load(f)
            json_cache = {'dataset': option, 'column': column_name,
                          'language_model': embedding_language, 'reduction_algorithm': transformer_option, 'sampled':sample_data}
            if cached_data != json_cache:
                my_bar = progress_bar.progress(0)
                with streamlit.spinner('Computing embeddings...'):
                    embedding_df = compute_to_cache(
                        embedding_language,
                        languages_dict,
                        transformer_option,
                        transformers_dict,
                        dataset,
                        option,
                        column_name,
                        my_bar,
                        sample_data)
                progress_bar.empty()
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
        if column_name != '-' and embedding_df is not None:

            try:

                if show_labeled:
                    temp_embedding_df = embedding_df.copy()
                else:
                    temp_embedding_df = embedding_df[embedding_df.labels == 'None']

                if suggest_clusters:

                    temp_embedding_df = cluster_suggestion.write(
                        temp_embedding_df, epsilon, min_samples, min_cluster_size, xi, options_container, xi_value)

                plot = make_interactive_plot(
                    temp_embedding_df, suggest_clusters)
                logging.info('plot has been shown')

                with chart_container:
                    result_lasso = streamlit_bokeh_events(
                        bokeh_plot=plot,
                        events="LASSO_SELECT",
                        key='hello',
                        refresh_on_update=True,
                        debounce_time=1)

                if result_lasso:

                    if result_lasso.get("LASSO_SELECT"):
                        try:
                            selected_data = temp_embedding_df.iloc[result_lasso.get("LASSO_SELECT")[
                                "data"]]
                        except:
                            selected_data = pd.DataFrame()

                        if len(selected_data != 0):
                            info_container.info(
                                f'selected {len(selected_data)} rows')
                            name_select_value = name_select.text_input(
                                'Input selected data label')
                            click_clear = click_clear_container.button(
                                'clear label')
                            if click_clear:
                                name_select_value = name_select.text_input(
                                    'Input selected data label', value='', key=1)
                                selected_data = pd.DataFrame()

                            if name_select_value:
                                selected_data['labels'] = name_select_value
                                try:
                                    temp_embedding_df.iloc[result_lasso.get("LASSO_SELECT")[
                                        "data"]] = selected_data
                                    embedding_df = replace_labels(
                                        embedding_df, temp_embedding_df, name_select_value)
                                    temp_embedding_df = embedding_df[embedding_df.labels == 'None']

                                except:
                                    pass

                                if not show_labeled:
                                    if suggest_clusters:

                                        temp_embedding_df = cluster_suggestion.write(
                                            temp_embedding_df, epsilon, min_samples, min_cluster_size, xi, options_container, xi_value)

                                    plot = make_interactive_plot(
                                        temp_embedding_df, suggest_clusters)

                                    with chart_container:
                                        result_lasso = streamlit_bokeh_events(
                                            bokeh_plot=plot,
                                            events="LASSO_SELECT",
                                            key='goodbye',
                                            refresh_on_update=True,
                                            debounce_time=1)

                                embedding_df.to_csv(
                                    'data/plotting_data/cache/cache.csv', index=False)

                                # wordcloud = generate_wordcloud(
                                #     selected_data.text.tolist())
                                # fig, ax = plt.subplots()
                                # wordcloud_fig = ax.imshow(
                                #     wordcloud, interpolation="bilinear")
                                # ax.axis('off')
                                # plt.savefig(
                                #     'data/plotting_data/cache/wordcloud.png')
                            if len(selected_data) != 0:
                                dataview_container.dataframe(
                                    selected_data.head(30), height=150)
                            else:
                                info_container.text('')

                            # wordcloud_container.image('data/plotting_data/cache/wordcloud.png',use_column_width=True)

            except Exception as error:
                streamlit.write(error)

    except:
        pass
