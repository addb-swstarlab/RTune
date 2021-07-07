# -*- coding: utf-8 -*-
"""
Train the model
"""

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
    
    print(wk_internal_metrics_data)