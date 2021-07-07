import os
import numpy as np


def load_knobs(k_path, db_name):
    if db_name == "redis":
        return redis_knobs_make_dict(k_path)
    elif db_name == "rocksdb":
        return rocksdb_knobs_make_dict(k_path)

def redis_knobs_make_dict(knobs_path):
    '''
        input: DataFrame form (samples_num, knobs_num)
        output: Dictionary form --> RDB and AOF
            ex. dict_knobs = {'columnlabels'=array([['knobs_1', 'knobs_2', ...],['knobs_1', 'knobs_2', ...], ...]),
                                'rowlabels'=array([1, 2, ...]),
                                'data'=array([[1,2,3,...], [2,3,4,...], ...[]])}

        For mode selection knob, "yes" -> 1 , "no" -> 0
    '''
    config_files = os.listdir(knobs_path)

    dict_RDB = {}
    dict_AOF = {}
    RDB_datas = []
    RDB_columns = []
    RDB_rowlabels = []
    AOF_datas = []
    AOF_columns = []
    AOF_rowlabels = []
    ISAOF = 0
    ISRDB = 1

    for m in range(len(config_files)):
        flag = 0
        datas = []
        columns = []
        knob_path = os.path.join(knobs_path, 'config'+str(m+1)+'.conf')
        f = open(knob_path, 'r')
        config_file = f.readlines()
        knobs_list = config_file[config_file.index('\n')+1:]

        cnt = 1

        for l in knobs_list:
            if l.split()[0] != 'save':
                col, d = l.split()
                if d.isalpha():
                    if d in ["no","yes"]:
                        d = ["no","yes"].index(d)
                    elif d in ["always","everysec","no"]:
                        d = ["always","everysec","no"].index(d)
                elif d.endswith("mb"):
                    d = d[:-2]
                datas.append(d)
                columns.append(col)
            else:
                col, d1, d2 = l.split()
                columns.append(col+str(cnt)+"_sec")
                columns.append(col+str(cnt)+"_changes")
                datas.append(d1)
                datas.append(d2)
                cnt += 1

            if l.split()[0] == 'appendonly':
                flag = ISAOF
            if l.split()[0] == 'save':
                flag = ISRDB

        # add active knobs
        if "activedefrag" not in columns:
            columns.append("activedefrag")
            # "0" means no
            datas.append("0")
            columns.append("active-defrag-threshold-lower")
            datas.append(10)
            columns.append("active-defrag-threshold-upper")
            datas.append(100)
            columns.append("active-defrag-cycle-min")
            datas.append(5)
            columns.append("active-defrag-cycle-max")
            datas.append(75)
        datas = list(map(int,datas))
        if flag == ISRDB:
    #         print('RDB')
            RDB_datas.append(datas)
            RDB_columns.append(columns)
            RDB_rowlabels.append(m+1)
        if flag == ISAOF: 
    #         print('AOF')
            AOF_datas.append(datas)
            AOF_columns.append(columns)
            AOF_rowlabels.append(m+1)

    dict_RDB['data'] = np.array(RDB_datas)
    dict_RDB['rowlabels'] = np.array(RDB_rowlabels)
    dict_RDB['columnlabels'] = np.array(RDB_columns[0])
    dict_AOF['data'] = np.array(AOF_datas)
    dict_AOF['rowlabels'] = np.array(AOF_rowlabels)
    dict_AOF['columnlabels'] = np.array(AOF_columns[0])
    return dict_RDB, dict_AOF

def rocksdb_knobs_make_dict(knobs_path):
    '''
        input: DataFrame form (samples_num, knobs_num)
        output: Dictionary form 
            ex. dict_knobs = {'columnlabels'=array([['knobs_1', 'knobs_2', ...],['knobs_1', 'knobs_2', ...], ...]),
                                'rowlabels'=array([1, 2, ...]),
                                'data'=array([[1,2,3,...], [2,3,4,...], ...[]])}

        For mode selection knob, "yes" -> 1 , "no" -> 0
    '''
    config_files = os.listdir(knobs_path)

    dict_data = {}
    datas = []
    columns = []
    rowlabels = []

    compression_type = ["snappy", "none", "lz4", "zlib"]
    cache_index_and_filter_blocks = ["false", "true"]

    for m in range(len(config_files)):
        flag = 0
        config_datas = []
        config_columns = []
        knob_path = os.path.join(knobs_path, 'config'+str(m+1)+'.cnf')
        f = open(knob_path, 'r')
        config_file = f.readlines()
        knobs_list = config_file[1:-1]

        for l in knobs_list:
            col, _, d = l.split()
            if d in compression_type:
                d = compression_type.index(d)
            elif d in cache_index_and_filter_blocks:
                d = cache_index_and_filter_blocks.index(d)
            config_datas.append(d)
            config_columns.append(col)

        datas.append(config_datas)
        columns.append(config_columns)
        rowlabels.append(m+1)

    dict_data['data'] = np.array(datas)
    dict_data['rowlabels'] = np.array(rowlabels)
    dict_data['columnlabels'] = np.array(columns[0])
    return dict_data
