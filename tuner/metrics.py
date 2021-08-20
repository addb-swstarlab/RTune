from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def load_metrics(m_path = ' ', labels = [], metrics=None, mode = ' ', target_wk = None, b = None):
    if mode == "internal":
        pd_metrics = pd.read_csv(m_path)
        pd_metrics, dict_le = metric_preprocess(pd_metrics)
        return metrics_make_dict(pd_metrics, labels), dict_le
    else:
        pd_metrics = pd.read_csv(m_path, index_col=0)
        #pd_metrics, dict_le = metric_preprocess(pd_metrics)
        if metrics == "SCORE":
            ## using pre-gained workloads
            if target_wk < 16:
                default_ex = [[19.5, 5.32, 10.6, 56.84], [53.6, 9.7, 9.5, 235.04], [104.9, 8.92, 10.4, 354.46], [131.6, 7.9, 11.3, 377.66], [8.1, 12.68, 8, 55.21],
                            [47.3, 10.88, 9.9, 228.18], [111, 8.34, 11.3, 344.46], [136.2, 7.55, 11.4, 366.74], [5, 20.51, 6.3, 53.94], [39.7, 12.92, 9.6, 222.84],
                            [123.6, 7.46, 11.8, 336.52], [133.9, 7.65, 12.2, 358.47], [4.5, 22.45, 6.2, 53.63], [31.5, 16.23, 8.2, 221.71], [99.3, 9.29, 10.5, 334.9], 
                            [112.6, 9.09, 10.7, 356.85]]
                default_ex_wk = default_ex[target_wk]
            ## using new target workloads
            elif target_wk > 15:
                default_ex = pd.read_csv('target_workload/default_performance/default_performance.csv', index_col=0)
                default_ex_wk = list(default_ex.loc[target_wk].values)
            
            pd_metrics[metrics] = default_ex_wk[0]/pd_metrics['TIME']*b[0]+pd_metrics['RATE']/default_ex_wk[1]*b[1] \
                                    +default_ex_wk[2]/pd_metrics['WAF']*b[2]+default_ex_wk[3]/pd_metrics['SA']*b[3]
            # return metrics_make_dict(pd_metrics, labels), None 
        return metrics_make_dict(pd_metrics[[metrics]], labels), None

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