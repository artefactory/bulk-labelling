from spacy import language
import streamlit
import os
import pandas as pd
import json
from streamlit_bokeh_events import streamlit_bokeh_events
import logging
import matplotlib.pyplot as plt


from lib.utils.load_config import load_config
from lib.utils.data_ingestion import load_dataset
from lib.utils.cache import (
    compute_cache,
    export_cache,
    clear_cache,
)
from lib.utils.plotting import make_interactive_plot, generate_wordcloud
from lib.utils.processing import replace_labels
from lib.utils import cluster_suggestion


def write():

    ###########################
    #     LOAD CONFIG         #
    ###########################

    (
        languages_dict,
        transformers_dict,
        datasets_dict,
        Embedding_frameworks_dataframe,
        save_path,
    ) = load_config()

    plot = None
    column_name = "-"
    dataset_language = None

    ##########################
    #      SIDEBAR           #
    ##########################

    streamlit.sidebar.title("Bulk labelling")

    ##########################
    #   UPLOADING DATASET    #
    ##########################

    dataset_upload = streamlit.sidebar.beta_expander("1. Select your dataset")

    option_box = dataset_upload.empty()
    dataframe_preview = dataset_upload.empty()
    uploaded_file = dataset_upload.file_uploader("Add a custom dataset")
    uploaded_file_name = dataset_upload.text_input("Custom dataset name", value="")

    dataset, option = load_dataset(
        uploaded_file, uploaded_file_name, datasets_dict, option_box
    )

    try:
        dataframe_preview.dataframe(dataset.head())
    except Exception:
        pass

        ##########################
        #   COLUMN SELECTION     #
        ##########################

    column_select = streamlit.sidebar.beta_expander("2. Select column for analysis")

    if dataset is not None:
        column_name = column_select.selectbox(
            "columns", options=["-"] + dataset.columns.tolist()
        )
        
        sample_data = column_select.checkbox("Sample a smaller dataset")
        remove_stopwords = column_select.checkbox("Remove stopwords")
        column_select.info(
            "One would sample the dataset to speed up calculations for a pre-labelling exploration phase"
        )
        if remove_stopwords:
            dataset_language = column_select.selectbox(
                "dataset language", ["French", "English"]
            )
            custom_stopwords = column_select.text_input(
                "Additional custom stopwords (comma-separated)"
            )
            if custom_stopwords is not None:
                custom_stopwords = (
                    pd.Series(custom_stopwords.split(","))
                    .apply(lambda x: x.replace(" ", ""))
                    .tolist()
                )
        else:
            custom_stopwords=None

        ##################################
        #   LANGUAGE MODEL SELECTION     #
        ##################################

    embedding = streamlit.sidebar.beta_expander("3. Select your embedding framework")
    embedding_lang_select = embedding.beta_container()
    embedding_lang = embedding.beta_container()
    languages_embedding = embedding_lang_select.multiselect(
        "Embedding framework languages",
        ["english", "french", "multilingual"],
        ["english", "french", "multilingual"],
    )
    embedding_language = embedding_lang.selectbox(
        "Embedding framework",
        Embedding_frameworks_dataframe[
            Embedding_frameworks_dataframe.language.isin(languages_embedding)
        ].framework.tolist(),
    )

    #######################################
    #   DIMENSION REDUCTION SELECTION     #
    #######################################

    map = streamlit.sidebar.beta_expander("4. Dimension reduction algorithm")
    transformer_option = map.selectbox(
        "Dimension reduction framework", ("TSNE", "PCA", "Umap")
    )

    #######################################
    #           CLUSTER SUGGESTION        #
    #######################################

    cluster_suggestion_sidebar = streamlit.sidebar.beta_expander(
        "5. Cluster suggestion"
    )

    suggest_clusters = cluster_suggestion_sidebar.checkbox("Suggest clusters?")
    suggest_clusters_slider = cluster_suggestion_sidebar.beta_container()
    if suggest_clusters:
        xi = suggest_clusters_slider.checkbox("use xi method")
        epsilon = suggest_clusters_slider.slider("epsilon", 0.0, 10.0, 4.0, 0.1)
        min_samples = suggest_clusters_slider.slider("min_samples", 1, 100, 50, 1)
        min_cluster_size = suggest_clusters_slider.slider(
            "min_cluster_size", 1, 100, 30, 1
        )
        if xi:
            xi_value = suggest_clusters_slider.slider("xi value", 0.0, 1.0, 0.05, 0.01)
        else:
            xi_value = 0.01


    ##################################
    #      EXPORTING DATA            #
    ##################################

    export = streamlit.sidebar.beta_expander("6. Export your labeled data")
    dataset_name_container = export.empty()
    dataset_name = dataset_name_container.text_input("enter desired name for file")
    click_clear_dataset_name = export.button("Next label", key="export dataset")
    if click_clear_dataset_name:
        dataset_name = dataset_name_container.text_input(
            "enter desired name for file", value="", key="empty dataset name"
        )
    export.info(
        "Exported data is saved by default to data/labeled_data/your_name.csv. modify this in the config file"
    )

        ##############################
        #       VARIOUS BUTTONS      #
        ##############################

    space = streamlit.sidebar.text("")
    compute = streamlit.sidebar.button("Compute embeddings")
    clear_cache_button = streamlit.sidebar.button("Clear cache")
    clear_labels_button = streamlit.sidebar.button("Clear labels")

    show_labeled = streamlit.sidebar.checkbox("Show labeled data")
    
    generate_sample_wordcloud = streamlit.sidebar.checkbox("Generate wordcloud")


    if clear_cache_button:
        clear_cache()

    ##########################
    #   PAGE STRUCTURE       #
    ##########################

    big_container = streamlit.beta_container()
    progress_indicator = big_container.beta_container()
    progress_bar = progress_indicator.empty()
    chart_container_over, options_container = big_container.beta_columns(2)
    chart_container = chart_container_over.empty()
    info_container = options_container.empty()
    dataview_container = options_container.empty()
    name_select = options_container.empty()
    click_clear_container = options_container.empty()

    #####################################
    #       COMPUTING AND DISPLAY       #
    #####################################

    if column_name != "-":

        dataset[column_name]=dataset[column_name].astype(str)
        embedding_df = compute_cache(
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
        )

        if embedding_df is not None:

            embedding_df = embedding_df.rename(
                columns={
                    embedding_df.columns[0]: "text",
                    embedding_df.columns[1]: "d1",
                    embedding_df.columns[2]: "d2",
                }
            )

            if "labels" not in embedding_df.columns or clear_labels_button:
                embedding_df["labels"] = "None"
                embedding_df.to_csv("data/plotting_data/cache/cache.csv", index=False)

            if dataset_name:
                export_cache(dataset_name, save_path)

    try:
        if column_name != "-" and embedding_df is not None:

            try:

                if show_labeled:
                    temp_embedding_df = embedding_df.copy()
                else:
                    temp_embedding_df = embedding_df[embedding_df.labels == "None"]

                if suggest_clusters:

                    temp_embedding_df = cluster_suggestion.write(
                        temp_embedding_df,
                        epsilon,
                        min_samples,
                        min_cluster_size,
                        xi,
                        options_container,
                        xi_value,
                    )

                plot = make_interactive_plot(temp_embedding_df, suggest_clusters)
                logging.info("plot has been shown")

                with chart_container:
                    result_lasso = streamlit_bokeh_events(
                        bokeh_plot=plot,
                        events="LASSO_SELECT",
                        key="hello",
                        refresh_on_update=True,
                        debounce_time=1,
                    )

                if result_lasso:

                    if result_lasso.get("LASSO_SELECT"):
                        try:
                            selected_data = temp_embedding_df.iloc[
                                result_lasso.get("LASSO_SELECT")["data"]
                            ]
                        except:
                            selected_data = pd.DataFrame()

                        if len(selected_data != 0):
                            info_container.info(f"selected {len(selected_data)} rows")
                            name_select_value = name_select.text_input(
                                "Input selected data label"
                            )
                            click_clear = click_clear_container.button("Next label")
                            if click_clear:
                                name_select_value = name_select.text_input(
                                    "Input selected data label", value="", key=1
                                )
                                selected_data = pd.DataFrame()

                            if name_select_value:
                                selected_data["labels"] = name_select_value
                                try:
                                    temp_embedding_df.iloc[
                                        result_lasso.get("LASSO_SELECT")["data"]
                                    ] = selected_data
                                    embedding_df = replace_labels(
                                        embedding_df,
                                        temp_embedding_df,
                                        name_select_value,
                                    )


                                    temp_embedding_df = embedding_df[
                                        embedding_df.labels == "None"
                                    ]

                                except Exception as e:
                                    streamlit.write(e)

                                if show_labeled:
                                    temp_embedding_df = embedding_df.copy()
                                else:
                                    temp_embedding_df = embedding_df[
                                        embedding_df.labels == "None"
                                    ]

                                if suggest_clusters:

                                    temp_embedding_df = cluster_suggestion.write(
                                        temp_embedding_df,
                                        epsilon,
                                        min_samples,
                                        min_cluster_size,
                                        xi,
                                        options_container,
                                        xi_value,
                                    )

                                plot = make_interactive_plot(
                                    temp_embedding_df, suggest_clusters
                                )

                                with chart_container:
                                    result_lasso = streamlit_bokeh_events(
                                        bokeh_plot=plot,
                                        events="LASSO_SELECT",
                                        key="goodbye",
                                        refresh_on_update=True,
                                        debounce_time=1,
                                    )

                                embedding_df.to_csv(
                                    "data/plotting_data/cache/cache.csv", index=False
                                )

                            if len(selected_data) != 0:
                                dataview_container.dataframe(
                                    selected_data[["text", "labels"]].head(30),
                                    height=150,
                                )
                            else:
                                info_container.text("")
                            if generate_sample_wordcloud:
                                wordcloud = generate_wordcloud(
                                    selected_data.text.tolist()
                                )
                                fig, ax = plt.subplots()
                                wordcloud_fig = ax.imshow(
                                    wordcloud, interpolation="bilinear"
                                )
                                ax.axis("off")
                                plt.savefig("data/plotting_data/wordcloud.png")

                                options_container.image(
                                    "data/plotting_data/wordcloud.png",
                                    use_column_width=True,
                                )

            except Exception as error:
                streamlit.write(error)

    except:
        pass
