# -*- coding: utf-8 -*-
"""
Train the model
"""

from operator import index
import os
import sys
import utils
import argparse
sys.path.append('../')
import copy
import numpy as np
import pandas as pd

from models.steps import (run_workload_characterization, run_knob_identification, configuration_recommendation)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tencent', action='store_true', help='Use Tencent Server')
    # parser.add_argument('--params', type=str, default='', help='Load existing parameters')
    # parser.add_argument('--target', type=int, default= 1, help='Workload type')    
    parser.add_argument('--persistence', type=str, choices=["RDB","AOF"], default='RDB', help='Choose Persistant Methods')
    parser.add_argument("--db",type=str, choices=["redis","rocksdb"], default='rocksdb', help="DB type")
    parser.add_argument("--exmetric", type=str, choices=["TIME", "RATE", "WAF", "SA"], default='WAF', help='Choose External Metrics')
    parser.add_argument("--rki",type=str, default='lasso', help = "knob_identification mode")
    parser.add_argument("--gp", type=str, default="numpy")
    parser.add_argument("--target", type=int, default=15, help="Choose which workload will be tagrget dataset")
    

    opt = parser.parse_args()

    DATA_PATH = "../data/{}_data".format(opt.db)
    
    PATH=None

    if not os.path.exists('logs'):
        os.mkdir('logs')

    if not os.path.exists('save_knobs'):
        os.mkdir('save_knobs')

    expr_name = 'train_{}'.format(utils.config_exist(opt.persistence, opt.db))


    print("======================MAKE LOGGER at %s====================" % expr_name)    
    logger = utils.Logger(
        name=opt.db,
        log_file='logs/{}/{}.log'.format(opt.persistence, expr_name) if opt.db == "redis" else 'logs/{}.log'.format(expr_name)
    )

    #==============================Data PreProcessing Stage=================================
    # Read sample-metric matrix, need knob name(label)
    '''
        internal_metrics, external_metrics, knobs
        metric_data : internal metrics
        knobs_data : configuration knobs
        ex. data = {'columnlabels'=array(['metrics_1', 'metrics_2', ...]),
                    'rowlabels'=array([1, 2, ...]),
                    'data'=array([[1,2,3,...], [2,3,4,...], ...[]])}
    '''

    wk_internal_metrics_path = os.path.join(DATA_PATH, 'results', 'internal')
    wk_external_metrics_path = os.path.join(DATA_PATH, 'results', 'external')

    logger.info("####################Target workload name is {}".format(opt.target))

    knobs_path = os.path.join(DATA_PATH, "configs")

    if opt.db == "redis":
        RDB_knob_data, AOF_knob_data = utils.load_knobs(knobs_path, opt.db)
        if opt.persistence == "RDB":
            knob_data = RDB_knob_data
        elif opt.persistence == "AOF":
            knob_data = AOF_knob_data
    elif opt.db == "rocksdb":
        knob_data = utils.load_knobs(knobs_path, opt.db)


    logger.info("Fin Load Knob_data")

    # train_knob_data = {}
    # test_knob_data = {}
    # train_internal_data = {}
    # test_internal_data = {}
    # train_external_data = {}
    # test_external_data = {}
    
    '''
        data format
        wk_internal,external_metrics_data is a list that includes all of workloads internal and external metrics data
            (workload_num, data_size, external(internal)_size) ex. (16, 20000, 1)
        knob_data is data of knobs
            (data_size, knob_size) ex. (20000, 22)
    '''
    
    wk_internal_metrics_data = []
    wk_external_metrics_data = []

    for wk in range(len(os.listdir(wk_internal_metrics_path))):
        wk_internal_metric, _ = utils.load_metrics(m_path = os.path.join(wk_internal_metrics_path, "internal_results_"+str(wk)+".csv"),
                                                labels = knob_data['rowlabels'],
                                                mode = 'internal')
        wk_internal_metrics_data.append(wk_internal_metric)

    logger.info("Fin Load internal_metrics_data")

    for wk in range(len(os.listdir(wk_external_metrics_path))):
        wk_external_metric, _ = utils.load_metrics(m_path = os.path.join(wk_external_metrics_path, "external_results_"+str(wk)+".csv"),
                                                labels = knob_data['rowlabels'],
                                                metrics = [opt.exmetric],
                                                mode = 'external')
        wk_external_metrics_data.append(wk_external_metric)                                                
    logger.info("Fin Load external_metrics_data")
    
    ## TODO: data split to train and test, test workload number is opt.target 
    # train_internal_data ==> 'rowlables' = (300000, ), 'data' = (300000, internal_size)
    # test_internal_data = wk_internal_metrics_data[opt.target]
    # train_external_data ==> 'rowlables' = (300000, ), 'data' = (300000, external_size) = (300000, 1)
    # test_external_data = wk_external_metrics_data[opt.target]
    train_internal_data = {'columnlabels':wk_internal_metrics_data[0]['columnlabels'],
                            'rowlabels':[i for i in range(1,300001)],
                            'data':[]}

    train_external_data = {'columnlabels':wk_external_metrics_data[0]['columnlabels'],
                            'rowlabels':[i for i in range(1,300001)],
                            'data':[]}                      

    for wk in range(len(wk_internal_metrics_data)):
        if wk+1 != opt.target:
            train_internal_data['data'].extend(wk_internal_metrics_data[wk]['data'])
            train_external_data['data'].extend(wk_external_metrics_data[wk]['data'])
    
    train_internal_data['data'] = np.array(train_internal_data['data'])
    train_external_data['data'] = np.array(train_external_data['data'])

    test_internal_data = wk_internal_metrics_data[opt.target]
    test_external_data = wk_external_metrics_data[opt.target]

    ## TODO: data split to train and test, test workload number is opt.target 
    # train_internal_data ==> 'rowlables' = (300000, ), 'data' = (300000, internal_size)
    # test_internal_data = wk_internal_metrics_data[opt.target]
    # train_external_data ==> 'rowlables' = (300000, ), 'data' = (300000, external_size) = (300000, 1)
    # test_external_data = wk_external_metrics_data[opt.target]
    

    ### METRICS SIMPLIFICATION STAGE ###
    """
        For example,
            pruned_metrics : ['allocator_rss_bytes', 'rss_overhead_bytes', 'used_memory_dataset', 'rdb_last_cow_size']
    """
    ## TODO: workload_characterization with train data set
    logger.info("\n\n====================== run_workload_characterization ====================")
    pruned_metrics = run_workload_characterization(train_internal_data)
    logger.info("Done pruning metrics for workload {} (# of pruned metrics: {}).\n\n""Pruned metrics: {}\n".format(opt.persistence, len(pruned_metrics),pruned_metrics))
    metric_idxs = [i for i, metric_name in enumerate(train_internal_data['columnlabels']) if metric_name in pruned_metrics]
    ranked_metric_data = {
        'data' : train_internal_data['data'][:,metric_idxs],
        'rowlabels' : copy.deepcopy(train_internal_data['rowlabels']),
        'columnlabels' : [train_internal_data['columnlabels'][i] for i in metric_idxs]
    }

    target_data = {
        'data' : test_internal_data['data'][:,metric_idxs],
        'rowlabels' : copy.deepcopy(test_internal_data['rowlabels']),
        'columnlabels' : [test_internal_data['columnlabels'][i] for i in metric_idxs]
    }

    print(type(target_data['data']))
    print(type(target_data['rowlabels']))
    print(type(target_data['columnlabels']))

    print(len(target_data['data']))             # len(['data'][0]) = 20
    print(len(target_data['rowlabels']))        # 20000
    print(len(target_data['columnlabels']))     # 20

    df_target = pd.DataFrame(target_data['data'], columns=target_data['columnlabels'], index=target_data['rowlabels'])
    df_internal = pd.DataFrame(ranked_metric_data['data'], columns=ranked_metric_data['columnlabels'], index=ranked_metric_data['rowlabels'])
    df_external = pd.DataFrame(train_external_data['data'], columns=train_external_data['columnlabels'], index=train_external_data['rowlabels'])

    print(df_target.head())
    print(df_internal.head())
    print(df_external.head())
    
    
    # df = pd.DataFrame(ranked_metric_data['data'][0], columns=ranked_metric_data['rowlabels'][0], index=ranked_metric_data['columnlabels'][0])
    # print(df)

    ### KNOBS RANKING STAGE ###
    ## TODO: workload_characterization with train data set
    rank_knob_data = copy.deepcopy(knob_data)
    logger.info("\n\n====================== run_knob_identification ====================")
    logger.info("use mode = {}".format(opt.rki))
    ranked_knobs = run_knob_identification(knob_data = rank_knob_data,
                                            metric_data = ranked_metric_data,
                                            mode = opt.rki,
                                            logger = logger)
    logger.info("Done ranking knobs for workload {} (# ranked knobs: {}).\n\n"
                 "Ranked knobs: {}\n".format(opt.persistence, len(ranked_knobs), ranked_knobs))

    




    ### COMBINED WORKLOAD GENERATION STAGE ###
    ## TODO: get Mahalanobis distance with statistic of each workload
    ##       pruning workload data using previous step's results
    ## sv_workload = extract_statistical_value(workload)
    ## mhd_workload = get_Mahalanobis_distance(sv_workload, target)
    ## ranked_mhd_workload = sort(mhd_workload)
    ## combined_workload = generation_combined_workload(ranked_mhd_workload, knob_data, external_metrics_data)
    nw = utils.test(opt.target, 50, df_internal, df_external)

    print(nw.head())

    assert False

    ### RECOMMENDATION STAGE ###
    ## TODO: genetic algorithm ...



    # top_ks = range(4,13)
    # best_recommend = -float('inf')
    # best_topk = None
    # best_conf_map = None
    # for top_k in top_ks:        
    #     logger.info("\n\n================ The number of TOP knobs ===============")
    #     logger.info(top_k)

    #     ranked_test_knob_data = utils.get_ranked_knob_data(ranked_knobs, test_knob_data, top_k)
        
    #     ## TODO: params(GP option) and will offer opt all
    #     FIN,recommend,conf_map = configuration_recommendation(ranked_test_knob_data,test_external_data, logger, opt.gp, opt.db, opt.persistence)

    #     if recommend > best_recommend and FIN:
    #         best_recommend = recommend
    #         best_topk = top_k
    #         best_conf_map = conf_map
    # logger.info("Best top_k")
    # logger.info(best_topk)
    # print(best_topk)
    # utils.convert_dict_to_conf(best_conf_map, opt.persistence)

    # print("END TRAIN")