from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def load_metrics(m_path = ' ', labels = [], metrics=None, mode = ' '):
    if mode == "internal":
        pd_metrics = pd.read_csv(m_path)
        pd_metrics, dict_le = metric_preprocess(pd_metrics)
        return metrics_make_dict(pd_metrics, labels), dict_le
    else:
        pd_metrics = pd.read_csv(m_path)
        #pd_metrics, dict_le = metric_preprocess(pd_metrics)
        return metrics_make_dict(pd_metrics[metrics], labels), None

def metric_preprocess(metrics):
    '''To invert for categorical internal metrics'''
    dict_le = {}
    c_metrics = metrics.copy()

    for col in metrics.columns:
        if isinstance(c_metrics[col][0], str):
            le = LabelEncoder()
            c_metrics[col] = le.fit_transform(c_metrics[col])
            dict_le[col] = le
    return c_metrics, dict_le

def metrics_make_dict(pd_metrics, labels):
    '''
        input: DataFrame form (samples_num, metrics_num)
        output: Dictionary form
            ex. dict_metrics = {'columnlabels'=array([['metrics_1', 'metrics_2', ...],['metrics_1', 'metrics_2', ...], ...]),
                            'rowlabels'=array([1, 2, ...]),
                            'data'=array([[1,2,3,...], [2,3,4,...], ...[]])}
    '''
    # labels = RDB or AOF rowlabels
    
    dict_metrics = {}
    tmp_rowlabels = [_-1 for _ in labels]
    pd_metrics = pd_metrics.iloc[tmp_rowlabels][:]
    nan_columns = pd_metrics.columns[pd_metrics.isnull().any()]
    pd_metrics = pd_metrics.drop(columns=nan_columns)
    
    # for i in range(len(pd_metrics)):
    #     columns.append(pd_metrics.columns.to_list())
    dict_metrics['columnlabels'] = np.array(pd_metrics.columns)
    #dict_metrics['columnlabels'] = np.array(itemgetter(*tmp_rowlabels)(dict_metrics['columnlabels'].tolist()))
    dict_metrics['rowlabels'] = np.array(labels)
    dict_metrics['data'] = np.array(pd_metrics.values)
    
    return dict_metrics