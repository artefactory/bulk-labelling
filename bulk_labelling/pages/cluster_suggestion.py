import streamlit
from preshed.maps import PreshMap
from cymem.cymem import Pool
from sklearn.cluster import OPTICS


from bulk_labelling.plotting import suggest_clusters,suggestion_chart

def write(embset):
    chart_container,options_container=streamlit.beta_columns(2)
    xi=options_container.checkbox('use xi method')
    epsilon=options_container.slider('epsilon',0.0,10.0,4.0,0.1)
    min_samples=options_container.slider('min_samples',1,100,50,1)
    min_cluster_size=options_container.slider('min_cluster_size',1,100,30,1)
    if xi:
        xi_value=options_container.slider('xi value',0.0,1.0,0.05,0.01)

    if xi:
        algo = OPTICS(cluster_method='xi',eps=epsilon,min_samples=min_samples,min_cluster_size=min_cluster_size,xi=xi_value)
    else:
        algo = OPTICS(cluster_method='dbscan',eps=epsilon,min_samples=min_samples,min_cluster_size=min_cluster_size)
    
    df = suggest_clusters(embset, algo)
    view_cluster=options_container.selectbox('Select cluster to view:',['all']+df.labels.unique().tolist())
    if view_cluster=='all':
        data=df.copy()
    else:
        data=df.copy()
        data.labels=data.labels.apply(lambda x:1 if x==view_cluster else 0)
    chart=suggestion_chart(data)
    chart_container.altair_chart(chart,use_container_width=True)