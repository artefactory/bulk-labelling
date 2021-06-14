from sklearn.cluster import OPTICS
from lib.utils.processing import cluster, cluster_tfidf
import streamlit

def write(embset, epsilon, min_samples, min_cluster_size, xi, options_container, xi_value):
    """generates an autocluster suggestion from parameters specified by the user and writes the resulting selectbox to the streamlit page.

    Args:
        embset (dataframe): dataframe containing features to cluster by as well as labels
        epsilon (float): epsilon value for both the DBScan and XI method algorithm
        min_samples (float): min_samples value for the OPTICS algorithm
        min_cluster_size (float): min_cluster_size value for the OPTICS algorithm
        xi (bool): switch to use XI algorithm or not
        options_container (streamlit container): container to write the cluster view selectbox in
        xi_value (float): xi value for both the DBScan and XI method algorithm

    Returns:
        dataframe: dataframe containing the features, labels and cluster suggestions from the clustering algo
    """
    if xi:
        algo = OPTICS(cluster_method='xi', eps=epsilon, min_samples=min_samples,
                      min_cluster_size=min_cluster_size, xi=xi_value)
    else:
        algo = OPTICS(cluster_method='dbscan', eps=epsilon,
                      min_samples=min_samples, min_cluster_size=min_cluster_size)

    df = cluster(algo, embset)

    df=cluster_tfidf(df)


    view_cluster = options_container.selectbox('Select cluster to view:', [
                                               'all']+df.cluster.unique().tolist(), key='selectbox')
    if view_cluster == 'all':
        data = df.copy()
    else:
        data = df.copy()
        data.cluster = data.cluster.apply(
            lambda x: '1' if x == view_cluster else '0')
    return data
