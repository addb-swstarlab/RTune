# -*- coding: utf-8 -*-

import time, os
import pickle, json
import logging
import datetime
from numpy.core.fromnumeric import compress
import pandas as pd
import numpy as np
from operator import itemgetter

import logging
from scipy.spatial import distance

def get_filename(PATH, head, tail):
    i = 0
    today = datetime.datetime.now()
    today = today.strftime('%Y%m%d')
    if not os.path.exists(os.path.join(PATH, today)):
        os.mkdir(os.path.join(PATH, today))
    name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    while os.path.exists(os.path.join(PATH, name)):
        i += 1
        name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    return name

def get_logger(log_path='./logs'):

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = logging.getLogger()
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('%(asctime)s[%(levelname)s] %(filename)s:%(lineno)s  %(message)s', date_format)
    name = get_filename(log_path, 'log', '.log')
    # i = 0
    # today = datetime.datetime.now()
    # name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    # while os.path.exists(os.path.join(log_path, name)):
    #     i += 1
    #     name = 'log-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.log'
    
    fileHandler = logging.FileHandler(os.path.join(log_path, name))
    streamHandler = logging.StreamHandler()
    
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    logger.setLevel(logging.INFO)
    logger.info('Writing logs at {}'.format(os.path.join(log_path, name)))
    return logger, os.path.join(log_path, name)

def time_start():
    return time.time()


def time_end(start):
    end = time.time()
    delay = end - start
    return delay


def get_timestamp():
    """
    获取UNIX时间戳
    """
    return int(time.time())


def time_to_str(timestamp):
    """
    将时间戳转换成[YYYY-MM-DD HH:mm:ss]格式
    """
    return datetime.datetime.\
        fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

class Logger:

    def __init__(self, name, log_file=''):
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        date_format = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter('%(asctime)s[%(levelname)s] %(filename)s:%(lineno)s  %(message)s', date_format)
        sh = logging.StreamHandler()
        fh = logging.FileHandler(log_file)
        sh.setFormatter(formatter)
        fh.setFormatter(formatter)
        self.logger.addHandler(sh)
        self.logger.addHandler(fh)
        if len(log_file) > 0:
            self.log2file = True
        else:
            self.log2file = False
        

    def _write_file(self, msg):
        if self.log2file:
            with open(self.log_file, 'a+') as f:
                f.write(msg + '\n')

    def get_timestr(self):
        timestamp = get_timestamp()
        date_str = time_to_str(timestamp)
        return date_str

    def warn(self, msg):
        msg = "%s[WARN] %s" % (self.get_timestr(), msg)
        self.logger.warning(msg)
        self._write_file(msg)
        print(msg)

    def info(self, msg):
        msg = "%s[INFO] %s" % (self.get_timestr(), msg)
        #self.logger.info(msg)
        self._write_file(msg)
        print(msg)

    def error(self, msg):
        # msg = "%s[ERROR] %s" % (self.get_timestr(), msg)
        self.logger.error(msg)
        # self._write_file(msg)


def save_state_actions(state_action, filename):

    f = open(filename, 'wb')
    pickle.dump(state_action, f)
    f.close()



def convert_dict_to_conf(rec_config, persistence):
    f = open('../data/redis_data/init_config.conf', 'r')
    json_configs_path = '../data/redis_data/'+persistence+'_knobs.json'
    with open(json_configs_path, 'r') as j:
        json_configs = json.load(j)

    dict_config = {}
    for d in json_configs:
        dict_config[d['name']] = d['default']

    config_list = f.readlines()
    save_f = False
    categorical_knobs = ['appendonly', 'no-appendfsync-on-rewrite', 'aof-rewrite-incremental-fsync',
                         'aof-use-rdb-preamble', 'rdbcompression', 'rdbchecksum', 
                         'rdb-save-incremental-fsync', 'activedefrag', 'activerehashing']
    
    if persistence == "RDB":
        save_sec = []
        save_changes = []   
    
    
    for k in dict_config.keys():
        if k in rec_config.keys():
            dict_config[k] = rec_config[k]

        dict_config[k] = round(dict_config[k])
        
        if k in categorical_knobs:
            if k == "activerehashing":
                if dict_config[k] == 0: dict_config[k] = 'no'
                elif dict_config[k] >= 1 : dict_config[k] = 'yes'
            else:
                if dict_config[k] == 0: dict_config[k] = 'no'
                elif dict_config[k] == 1: dict_config[k] = 'yes'
        if k == 'appendfsync':
            if dict_config[k] == 0: dict_config[k] = 'always'
            elif dict_config[k] == 1: dict_config[k] = 'everysec'
            elif dict_config[k] >= 2: dict_config[k] = 'no'    

        if 'changes' in k or 'sec' in k:
            save_f = True
            if 'sec' in k:
                save_sec.append(dict_config[k])
            if 'changes' in k:
                save_changes.append(dict_config[k])
            continue
        
        if k == 'auto-aof-rewrite-min-size':
            dict_config[k] = str(dict_config[k]) + 'mb'

        config_list.append(k+' '+str(dict_config[k])+'\n')
    
    if save_f:
        for s in range(len(save_sec)):
            config_list.append('save ' + str(save_sec[s]) + ' ' + str(save_changes[s]) + '\n')
    i = 0
    PATH = '../data/redis_data/config_results/{}'.format(persistence)
    NAME = persistence+'_rec_config{}.conf'.format(i)
    while os.path.exists(os.path.join(PATH,NAME)):
        i+=1
        NAME = persistence+'_rec_config{}.conf'.format(i)
    
    with open(os.path.join(PATH,NAME), 'w') as rec_f:
        rec_f.writelines(config_list) 

def config_exist(persistence, db_name):
    i = 0
    PATH = '../data/{}_data/config_results/'.format(db_name)
    NAME = 'rec_config'
    if db_name == 'redis':
        PATH = PATH + persistence
        NAME = persistence + '_' + NAME
    CONFIG_NAME = NAME + '{}.conf'.format(i)
    while os.path.exists(os.path.join(PATH,CONFIG_NAME)):
        i+=1
        CONFIG_NAME = NAME + '{}.conf'.format(i)
    return CONFIG_NAME[:-5]



from sklearn.preprocessing import StandardScaler

# Modifying to import upper folder
import sys
sys.path.append('../')
from models.util import DataUtil


def process_training_data(target_knob, target_metric, db_type, data_type):
    # Load mapped workload data
    # TODO: If we have mapped_workload, we will use this code
    
    # if target_data['mapped_workload'] is not None:
        # mapped_workload_id = target_data['mapped_workload'][0]
        # mapped_workload = Workload.objects.get(pk=mapped_workload_id)
        # workload_knob_data = PipelineData.objects.get(
        #     pipeline_run=latest_pipeline_run,
        #     workload=mapped_workload,
        #     task_type=PipelineTaskType.KNOB_DATA)
        # workload_knob_data = JSONUtil.loads(workload_knob_data.data)
        # workload_metric_data = PipelineData.objects.get(
        #     pipeline_run=latest_pipeline_run,
        #     workload=mapped_workload,
        #     task_type=PipelineTaskType.METRIC_DATA)
        # workload_metric_data = JSONUtil.loads(workload_metric_data.data)
        # cleaned_workload_knob_data = DataUtil.clean_knob_data(workload_knob_data["data"],
        #                                                       workload_knob_data["columnlabels"],
        #                                                       [newest_result.session])
        # X_workload = np.array(cleaned_workload_knob_data[0])
        # X_columnlabels = np.array(cleaned_workload_knob_data[1])
        # y_workload = np.array(workload_metric_data['data'])
        # y_columnlabels = np.array(workload_metric_data['columnlabels'])
        # rowlabels_workload = np.array(workload_metric_data['rowlabels'])
    if False:
        pass
    else:
        # combine the target_data with itself is actually adding nothing to the target_data
        X_workload = np.array(target_knob['data'])
        X_columnlabels = np.array(target_knob['columnlabels'])
        y_workload = np.array(target_metric['data'])
        y_columnlabels = np.array(target_metric['columnlabels'])
        rowlabels_workload = np.array(target_knob['rowlabels'])

    # Target workload data
    X_target = target_knob['data']
    y_target = target_metric['data']
    rowlabels_target = np.array(target_knob['rowlabels'])

    if not np.array_equal(X_columnlabels, target_knob['columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical X columnlabels (sorted knob names)'),
                        X_columnlabels, target_knob['X_columnlabels'])
    if not np.array_equal(y_columnlabels, target_metric['columnlabels']):
        raise Exception(('The workload and target data should have '
                         'identical y columnlabels (sorted metric names)'),
                        y_columnlabels, target_metric['columnlabels'])

    # TODO: If we have mapped_workload, we will use this code
    # Filter ys by current target objective metric
    # target_objective = newest_result.session.target_objective
    # target_obj_idx = [i for i, cl in enumerate(y_columnlabels) if cl == target_objective]
    # if len(target_obj_idx) == 0:
    #     raise Exception(('Could not find target objective in metrics '
    #                      '(target_obj={})').format(target_objective))
    # elif len(target_obj_idx) > 1:
    #     raise Exception(('Found {} instances of target objective in '
    #                      'metrics (target_obj={})').format(len(target_obj_idx),
    #                                                        target_objective))

    # y_workload = y_workload[:, target_obj_idx]
    # y_target = y_target[:, target_obj_idx]
    # y_columnlabels = y_columnlabels[target_obj_idx]

    # y_workload = y_workload[:, 0]
    # y_target = y_target[:, 0]
    # y_columnlabels = y_columnlabels[0]

    # Combine duplicate rows in the target/workload data (separately)
    X_workload, y_workload, rowlabels_workload = DataUtil.combine_duplicate_rows(
        X_workload, y_workload, rowlabels_workload)
    X_target, y_target, rowlabels_target = DataUtil.combine_duplicate_rows(
        X_target, y_target, rowlabels_target)

    # Delete any rows that appear in both the workload data and the target
    # data from the workload data
    # dups_filter = np.ones(X_workload.shape[0], dtype=bool)
    # target_row_tups = [tuple(row) for row in X_target]
    # for i, row in enumerate(X_workload):
    #     if tuple(row) in target_row_tups:
    #         dups_filter[i] = False
    # X_workload = X_workload[dups_filter, :]
    # y_workload = y_workload[dups_filter, :]
    # rowlabels_workload = rowlabels_workload[dups_filter]

    # Combine target & workload Xs for preprocessing
    X_matrix = np.vstack([X_target,X_workload])

    dummy_encoder = None
    
    # Scale to N(0, 1)
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_matrix)
    if y_target.shape[0] < 5:  # FIXME
        # FIXME (dva): if there are fewer than 5 target results so far
        # then scale the y values (metrics) using the workload's
        # y_scaler. I'm not sure if 5 is the right cutoff.
        y_target_scaler = None
        y_workload_scaler = StandardScaler()
        y_matrix = np.vstack([y_target, y_workload])
        y_scaled = y_workload_scaler.fit_transform(y_matrix)
    else:
        # FIXME (dva): otherwise try to compute a separate y_scaler for
        # the target and scale them separately.
        try:
            y_target_scaler = StandardScaler()
            y_workload_scaler = StandardScaler()
            y_target_scaled = y_target_scaler.fit_transform(y_target)
            y_workload_scaled = y_workload_scaler.fit_transform(y_workload)
            y_scaled = np.vstack([y_target_scaled, y_workload_scaled])
        except ValueError:
            y_target_scaler = None
            y_workload_scaler = StandardScaler()
            y_scaled = y_workload_scaler.fit_transform(y_target)

    # Maximize the throughput, moreisbetter
    # If Use gradient descent to minimize -throughput
    # if not lessisbetter:
    #     y_scaled = -y_scaled

    # FIXME (dva): check if these are good values for the ridge
    # ridge = np.empty(X_scaled.shape[0])
    # ridge[:X_target.shape[0]] = 0.01
    # ridge[X_target.shape[0]:] = 0.1
    X_min = np.empty(X_scaled.shape[1])
    X_max = np.empty(X_scaled.shape[1])
    X_scaler_matrix = np.zeros([1, X_scaled.shape[1]])

    with open(os.path.join("../data/{}_data".format(db_type),data_type+"_knobs.json"), "r") as data:
        session_knobs = json.load(data)

    # Set min/max for knob values
    #TODO : we make binary_index_set
    for i in range(X_scaled.shape[1]):
        col_min = X_scaled[:, i].min()
        col_max = X_scaled[:, i].max()
        for knob in session_knobs:
            if X_columnlabels[i] == knob["name"]:
                if knob["minval"]==0:
                    col_min = knob["minval"]
                    col_max = knob["maxval"]
                else:
                    X_scaler_matrix[0][i] = knob["minval"]
                    col_min = X_scaler.transform(X_scaler_matrix)[0][i]
                    X_scaler_matrix[0][i] = knob["maxval"]
                    col_max = X_scaler.transform(X_scaler_matrix)[0][i]
            X_min[i] = col_min
            X_max[i] = col_max

    return X_columnlabels, X_scaler, X_scaled, y_scaled, X_max, X_min, dummy_encoder



# Distance
def extract_statistical_value(wk_internal_metrics):
    wk_stats_list = []
    drop_columns = ['count', 'min', 'max']

    for wk_internal_metric in wk_internal_metrics:
        wk_stats = wk_internal_metric.describe().T.drop(columns=drop_columns)
        wk_stats_list.append(wk_stats)
    return 


def get_rep(sampleSize, internal_metrics):

    df = []
    raw_df = []
    drop_columns = ['count', 'min', 'max']

    df_im = internal_metrics

    for i in range(16):
            df_im = internal_metrics.sample(sampleSize)
            # df_im = df_im.drop(columns=['index'])
            raw_df.append(df_im)
            df_im = df_im.describe().T.drop(columns=drop_columns)
            df.append(df_im.T)
            
    return df

def get_idx(df):
    return df[0].columns

def get_cov(target, df, df_idx):
        
    df_cov = {}

    for col in df[target].columns:
        im = []
        for l in range(len(df)):
            im.append(df[l][[col]].T)
        im = pd.concat(im)
        im_cov = im.cov()
        df_cov[col] = im_cov
        
    return df_cov

def get_mah(train, tar, df_cov, df_idx):
    mah_dis = {}
    sum_d = 0
    for i, idx in enumerate(df_idx):
        if np.linalg.det(df_cov[df_idx[i]]) != 0:
            d = distance.mahalanobis(u=train[idx], v=tar[idx], VI=np.linalg.pinv(df_cov[idx]))
            mah_dis[idx] = d
            sum_d += d
    return mah_dis, sum_d

def get_dist(df, df_idx, df_cov, target):

    d_list = dict()

    for wn in range(16):
        
        if (wn != target):

            _, s_d = get_mah(df[wn], df[target], df_cov, df_idx)

            d_list[wn]=s_d
    
    return d_list

def get_score(df, df_idx, df_cov, target):

    d_list = get_dist(df, df_idx, df_cov, target)

    score_list = dict()

    d_sum = sum(d_list.values())
    
    M = max(d_list.values())

    for k, v in d_list.items():
        score = M - v
        score_list[k] = score
    
    s_sum = sum(score_list.values())
    
    for k, v in score_list.items():
        score = v/s_sum
        
        score_list[k] = score
    
    return score_list

def test(target, sampleSize, internal_metrics, external_metrics):
    
    df = get_rep(sampleSize, internal_metrics)
    print("Get df")
    
    df_idx = get_idx(df)
    print("Get df_idx")
    
    df_cov = get_cov(target, df, df_idx)
    print("Get df_cov")
    
    
    score = sorted(get_score(df, df_idx, df_cov, target).items(), key=itemgetter(1), reverse=True)

    new_workload = pd.DataFrame() 

    for k, v in score:
        df_ex = external_metrics.sort_values(by=['WAF'], axis=0, ascending=True)

        target = int(20000*v)+1
        start = 0
        end = target

        while target > 0:

            t_workload = df_ex[start : end]
            t_workload["WL"] = k
            start = end

            totalNum = len(new_workload)
            new_workload = pd.concat([new_workload, t_workload])

            preDrop = len(new_workload)

            # new_workload = new_workload.drop_duplicates(['index'])

            postDrop = len(new_workload)

            dropCol = preDrop - postDrop
            addCol = postDrop - totalNum

            target = target - addCol
            end = end + target

            if( len(new_workload) >= 20000):
                break

            # 1. 전체에 target 만큼 추가
            # 2. 중복을 제거
            # 3. 제거된 숫자 count
            # 4. target에 제거된 숫자를 뺌
            # 5. target workload의 추가 이후부터 다시 1번

        if( len(new_workload) >= 20000):
            break

    new_workload = new_workload[:20000].sort_index()
    
    return new_workload