# -*- coding: utf-8 -*-
"""
Train the model
"""

import os
import sys, logging
import copy
import numpy as np
import pandas as pd
import argparse
import datetime
import utils
from knobs import load_knobs
from metrics import load_metrics
sys.path.append('../')
from models.steps import (run_workload_characterization, run_knob_identification, configuration_recommendation, get_ranked_knob_data, generation_combined_workload)

parser = argparse.ArgumentParser()
# parser.add_argument('--params', type=str, default='', help='Load existing parameters')
# parser.add_argument('--target', type=int, default= 1, help='Workload type')    
parser.add_argument('--persistence', type=str, choices=["RDB","AOF"], default='RDB', help='Choose Persistant Methods')
parser.add_argument("--db",type=str, choices=["redis","rocksdb"], default='rocksdb', help="DB type")
parser.add_argument("--exmetric", type=str, choices=["TIME", "RATE", "WAF", "SA", "SCORE"], default='RATE', help='Choose External Metrics')
parser.add_argument("--rki",type=str, default='lasso', help = "knob_identification mode")
parser.add_argument("--gp", type=str, default="numpy")
parser.add_argument("--target", type=int, default=16, help="Choose which workload will be tagrget dataset, 0~15 are pre-gained worklaods, 16~ is a new target workload in folder")
parser.add_argument("--targetsize", type=int, default=20, help="Define the number of target workload data")
parser.add_argument("--targetresultpath", type=str, default='target_workload/min_time_conf_result', help="Define the target workload path")
# related run_knob_identification
parser.add_argument("--topk", type=int, default=-1, help="Define k to prune knob ranking data") 
# related run_workload_characterization
parser.add_argument("--isskip", type=bool, default=True, help="Skip the workload characterization or not") 
parser.add_argument("--cluster", type=int, default=10, help="limit the number of cluster")
# related generation_combined_workload
parser.add_argument("--iscombined", type=bool, default=True, help="Combine the workloads or not") 
# related dense layer batch_size=64, epochs=300, lr=0.0001
parser.add_argument("--mode", type=str, default='dense', choices=['dense', 'multi'], help="Define which mode will use fitness function for GA in recommendation step")
parser.add_argument("--balance", type=list, default=[0.25, 0.25, 0.25, 0.25], help="Define balance number to calculate score")
parser.add_argument("--batch_size", type=int, default=64, help="Define batch size to train model")
parser.add_argument("--epochs", type=int, default=300, help="Define epochs to train model")
parser.add_argument("--lr", type=int, default=0.0002, help="Define learning rate to train model")
parser.add_argument("--pool", type=int, default=128, help="Define the number of pool to GA algorithm")
parser.add_argument("--generation", type=int, default=1000, help="Define the number of generation to GA algorithm")




opt = parser.parse_args()

DATA_PATH = "../data/{}_data".format(opt.db)

PATH=None

if not os.path.exists('logs'):
    os.mkdir('logs')

if not os.path.exists('save_knobs'):
    os.mkdir('save_knobs')

# expr_name = 'train_{}'.format(utils.config_exist(opt.persistence, opt.db))


print("======================MAKE LOGGER at====================")    
logger, log_dir = utils.get_logger(os.path.join('./logs'))
# logger = utils.Logger(
#     name=opt.db,
#     log_file='logs/{}/{}.log'.format(opt.persistence, expr_name) if opt.db == "redis" else 'logs/{}.log'.format(expr_name)
# )

def main():
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
    
    logger.info("Target workload name is {}, target external metric is {}".format(opt.target, opt.exmetric))

    knobs_path = os.path.join(DATA_PATH, "configs")

    if opt.db == "redis":
        RDB_knob_data, AOF_knob_data = load_knobs(knobs_path, opt.db)
        if opt.persistence == "RDB":
            knob_data = RDB_knob_data
        elif opt.persistence == "AOF":
            knob_data = AOF_knob_data
    elif opt.db == "rocksdb":
        knob_data = load_knobs(knobs_path, opt.db)

    logger.info("Fin Load Knob_data")

    '''
        ### GENERATE DICTIONARY DATA WITH EACH WORKLOAD ###
        data format
        wk_internal,external_metrics_data is a list that includes all of workloads internal and external metrics data
            (workload_num, data_size, external(internal)_size) ex. (16, 20000, 1)
        knob_data is data of knobs
            (data_size, knob_size) ex. (20000, 22)
    '''
    
    wk_internal_metrics_data = []
    wk_external_metrics_data = []

    for wk in range(len(os.listdir(wk_internal_metrics_path))):
        wk_internal_metric, _ = load_metrics(m_path = os.path.join(wk_internal_metrics_path, "internal_results_"+str(wk)+".csv"),
                                                labels = knob_data['rowlabels'],
                                                mode = 'internal')
        wk_internal_metrics_data.append(wk_internal_metric)

    logger.info("Fin Load internal_metrics_data")

    for wk in range(len(os.listdir(wk_external_metrics_path))):
        wk_external_metric, _ = load_metrics(m_path = os.path.join(wk_external_metrics_path, "external_results_"+str(wk)+".csv"),
                                                labels = knob_data['rowlabels'],
                                                metrics = opt.exmetric,
                                                mode = 'external',
                                                target_wk = opt.target,
                                                b = opt.balance)
        wk_external_metrics_data.append(wk_external_metric)                                                
    logger.info("Fin Load external_metrics_data")
    
    ### SPLIT DATA TO TRAIN AND TEST ###
    train_internal_data = {'columnlabels':wk_internal_metrics_data[0]['columnlabels'],
                            'rowlabels':[i for i in range(1,300001)],
                            'data':[]}

    train_external_data = {'columnlabels':wk_external_metrics_data[0]['columnlabels'],
                            'rowlabels':[i for i in range(1,300001)],
                            'data':[]}                      

    for wk in range(len(wk_internal_metrics_data)):
        if wk != opt.target:
            train_internal_data['data'].extend(wk_internal_metrics_data[wk]['data'])
            train_external_data['data'].extend(wk_external_metrics_data[wk]['data'])
    
    train_internal_data['data'] = np.array(train_internal_data['data'])
    train_external_data['data'] = np.array(train_external_data['data'])

    # test_internal_data = wk_internal_metrics_data[opt.target]
    # test_external_data = wk_external_metrics_data[opt.target]
    
    ### METRICS SIMPLIFICATION STAGE ###
    """
        For example,
            pruned_metrics : ['allocator_rss_bytes', 'rss_overhead_bytes', 'used_memory_dataset', 'rdb_last_cow_size']
    """
    logger.info("\n\n====================== run_workload_characterization ====================")
    pruned_metrics = run_workload_characterization(train_internal_data, cluster_threshold= opt.cluster, skip=opt.isskip)
    logger.info("Done pruning metrics for workload {} (# of pruned metrics: {}).\n\n""Pruned metrics: {}\n".format(opt.persistence, len(pruned_metrics),pruned_metrics))
    metric_idxs = [i for i, metric_name in enumerate(train_internal_data['columnlabels']) if metric_name in pruned_metrics]
    ranked_metric_data = {
        'data' : train_internal_data['data'][:,metric_idxs],
        'rowlabels' : copy.deepcopy(train_internal_data['rowlabels']),
        'columnlabels' : [train_internal_data['columnlabels'][i] for i in metric_idxs]
    }
    wk_pruned_internal_metrics_data = []
    for wk_int_met in wk_internal_metrics_data:
        wk_int_met['data'] = wk_int_met['data'][:,metric_idxs]
        wk_pruned_internal_metrics_data.append(wk_int_met)    

    ### KNOBS RANKING STAGE ###
    rank_knob_data = copy.deepcopy(knob_data)
    logger.info("\n\n====================== run_knob_identification ====================")
    logger.info("use mode = {}".format(opt.rki))
    top_k = opt.topk
    if opt.topk != -1:
        ranked_knobs = run_knob_identification(knob_data = rank_knob_data,
                                                metric_data = ranked_metric_data,
                                                mode = opt.rki,
                                                logger = logger)
        logger.info("Done ranking knobs for workload (# ranked knobs: {}).\n\n"
                    "Ranked knobs: {}\n".format(len(ranked_knobs), ranked_knobs))
    else:
        logger.info("Skipping ranking knobs")

    ### PRUNING KNOB DATA WITH TOP K ###
    
    logger.info("\n\n================ The number of TOP {} knobs ===============".format(top_k))
    if opt.topk != -1:
        ranked_test_knob_data = get_ranked_knob_data(ranked_knobs, knob_data, top_k)
        logger.info('Pruned Ranked knobs: {}'.format(ranked_test_knob_data['columnlabels']))
    else:
        logger.info("Skipping pruning ranked knobs")

    ### COMBINED WORKLOAD GENERATION STAGE ###
    ## TODO: get Mahalanobis distance with statistic of each workload
    ##       pruning workload data using previous step's results
    logger.info("\n\n====================== generation_combined_workload ====================")
    combined_wk_external_metrics_data = generation_combined_workload(wk_pruned_internal_metrics_data, wk_external_metrics_data, opt.target, 
                                                                     opt.targetsize, logger, opt.exmetric, opt.iscombined,
                                                                     opt.targetresultpath)
                                                                    
    # print(combined_wk_external_metrics_data)
    
    ## Save Combined Workload csv file
    combined_wk_PATH = 'combined_wk'
    head_name = 'cbwk_'+str(opt.target)
    tail_name = '.csv'
    name = utils.get_filename(combined_wk_PATH, head_name, tail_name)
    combined_wk_external_metrics_data.to_csv(os.path.join(combined_wk_PATH, name))
    logger.info(f"\n\n====================== save_combined_workload to {os.path.join(combined_wk_PATH, name)}====================")
    # i = 0
    # today = datetime.datetime.now()
    # name = 'cbwk_'+str(opt.target)+'-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.csv'
    # while os.path.exists(os.path.join(combined_wk_PATH, name)):
    #     i += 1
    #     name = 'cbwk_'+str(opt.target)+'-'+today.strftime('%Y%m%d')+'-'+'%02d'%i+'.csv'
    
    ### RECOMMENDATION STAGE ###
    ## TODO: genetic algorithm ...
    ## train dense layer model with combined worklaod ##
    logger.info("\n\n====================== train_combined_workload ====================")
    configuration_recommendation(knob_data, combined_wk_external_metrics_data, logger, mode=opt.mode,
                                 batch_size=opt.batch_size, epochs=opt.epochs, lr=opt.lr, n_pool=opt.pool, 
                                 n_generation=opt.generation, b=opt.balance)

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

if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()
        logging.shutdown()