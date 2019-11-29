"""
**Author** : Robin Camarasa 

**Institution** : Erasmus Medical Center

**Position** : PhD student

**Contact** : r.camarasa@erasmusmc.nl

**Date** : 2019-11-28

**Project** : data_science_framework

TODO: rewrite whole function
TODO: doc
**  **
"""


def log_to_csv(result_path, data):
    """


    :param result_path:
    :param data:
    :return:
    """
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    try:
        dataframe = pd.read_csv(os.path.join(result_path, 'dashboard.csv'), index_col='id')
        dataframe = dataframe.append(pd.Series(data, name=dataframe.index.max() + 1))
    except Exception as e:
        dataframe = pd.DataFrame(
            index=pd.Index([1], name='id'), data=[data],
        )
    dataframe.to_csv(os.path.join(result_path, 'dashboard.csv'))
    id = dataframe.index.max()
    os.makedirs(os.path.join(result_path, str(float(id))))
    return id


def local_log(result_path, function, data):
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    try:
        dataframe = pd.read_csv(
            os.path.join(result_path, function + '.csv')
        )
        dataframe = dataframe.append(pd.Series(data))
    except Exception as e:
        dataframe = pd.DataFrame(data=[data])
    dataframe.to_csv(os.path.join(result_path, function + '.csv'), index=False)


def make_log_dir(result_path='test', data='test'):
    """

    :param result_path:
    :param data:
    :return:
    """
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    try:
        dataframe = pd.read_csv(os.path.join(result_path, 'dashboard.csv'), index_col='id')
        dataframe = dataframe.append(pd.Series(data, name=dataframe.index.max() + 1))
    except Exception as e:
        dataframe = pd.DataFrame(
            index=pd.Index([1], name='id'), data=[data],
        )
    dataframe.to_csv(os.path.join(result_path, 'dashboard.csv'))
    id = dataframe.index.max()
    os.makedirs(os.path.join(result_path, str(float(id))))
    return id


