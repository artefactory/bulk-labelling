import os
import uuid
import pandas as pd


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

    """loads dataset, taking into account the sample datasets provided and potential user uploads

    Returns:
        pd.DataFrame:  dataset to analyze
    """
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

