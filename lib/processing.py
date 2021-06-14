def replace_labels(embedding_df, temp_embedding_df, label):
    """replaces the label in a given part of the embedding dataframe

    Args:
        embedding_df (pd.DataFrame): dataframe of dimension-reduced encodings for text
        temp_embedding_df (pd.DataFrame): temporary dataframe with the selected data and/or clustering
        label (str): label to replace the labels in embedding_df with

    Returns:
        pd.DataFrame(): dataframe with replaced labels where necessary
    """
    embedding_df.loc[embedding_df.labelling_uuid.isin(
        temp_embedding_df[temp_embedding_df.labels != 'None'].labelling_uuid), 'labels'] = label

    return embedding_df

def cluster(algo, embset):
    """suggests cluster based on provided algo and embedding dataframe

    Args:
        algo (sklearn.transformer): clustering algorithm used to cluster data
        embset (pd.DataFrame): embedding dataframe containing x,y data to cluster

    Returns:
        pd.dataFrame(): embedding dataframe with cluster suggestion values
    """
    transformed = embset[['d1', 'd2']].values
    clustering = algo.fit(transformed)
    labels = clustering.labels_
    embed = embset.copy()
    embed['cluster'] = labels
    embed['cluster'] = embed['cluster'].astype(str)
    return embed
