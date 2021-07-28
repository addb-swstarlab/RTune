import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os,sys
import pandas as pd
import numpy as np
import torch
from knobs import load_knobs
import utils
sys.path.append('../')
from models.denseGA import fitness_function
import models.rocksdb_option as option


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Define saved model')
parser.add_argument('--combined_wk', type=str, help='Define saved combined workload')
parser.add_argument("--pool", type=int, default=128, help="Define the number of pool to GA algorithm")
parser.add_argument("--generation", type=int, default=20000, help="Define the number of generation to GA algorithm")
parser.add_argument("--db",type=str, choices=["redis","rocksdb"], default='rocksdb', help="DB type")

if not os.path.exists('eval-logs'):
    os.mkdir('eval-logs')

print("======================MAKE LOGGER at====================")    
logger, log_dir = utils.get_logger(os.path.join('./eval-logs'))

opt = parser.parse_args()

DATA_PATH = "../data/{}_data".format(opt.db)
knobs_path = os.path.join(DATA_PATH, "configs")

best_model = torch.load(os.path.join('model_save', opt.model))
best_model.eval()

knob_data = load_knobs(knobs_path, opt.db)
configs = knob_data['data']

n_configs = configs.shape[1] # get number of 22
n_pool_half = int(opt.pool/2) # hafl of a pool size
mutation = int(n_configs * 0.4) # mutate percentage

knob_data = load_knobs(knobs_path, opt.db)
configs = knob_data['data']
combined_wk = pd.read_csv(os.path.join('combined_wk', opt.combined_wk), index_col=0)

X_tr, X_te, y_tr, y_te = train_test_split(configs, np.array(combined_wk), test_size=0.2, random_state=42, shuffle=True)
scaler_X = MinMaxScaler().fit(X_tr) # range: 0~1
scaler_y = StandardScaler().fit(y_tr)

current_solution_pool = configs[:opt.pool] # dataframe -> numpy

for i in range(opt.generation):
    ## data scaling
    scaled_pool = scaler_X.transform(current_solution_pool)
    ## fitenss calculation with real values (not scaled)
    scaled = fitness_function(scaled_pool, best_model)
    fitness = scaler_y.inverse_transform(scaled)
    ## best parents selection
    idx_fitness = np.argsort(fitness)[:n_pool_half] # minimum
    best_solution_pool = current_solution_pool[idx_fitness,:]
    if i % 1000 == 999:
        logger.info(f"[{i+1:3d}/{opt.generation:3d}] best fitness: {fitness[idx_fitness[0]]:.5f}")
    ## cross-over and mutation
    new_solution_pool = np.zeros_like(best_solution_pool) # new_solution_pool.shape = (n_pool_half,22)
    for j in range(n_pool_half):
        pivot = np.random.choice(np.arange(1,n_configs-1))
        new_solution_pool[j][:pivot] = best_solution_pool[j][:pivot]
        new_solution_pool[j][pivot:] = best_solution_pool[n_pool_half-1-j][pivot:]
        
        _, random_knobs = option.make_random_option()
        knobs = list(random_knobs.values())
        random_knob_index = np.arange(n_configs)
        np.random.shuffle(random_knob_index)
        random_knob_index = random_knob_index[:mutation]
        # random_knob_index = [random.randint(0,21) for r in range(mutation)]
        for k in range(len(random_knob_index)):
            new_solution_pool[j][random_knob_index[k]] = knobs[random_knob_index[k]]
    
    ## stack
    current_solution_pool = np.vstack([best_solution_pool, new_solution_pool]) # current_solution_pool.shape = (n_pool,22)
    
final_solution_pool = pd.DataFrame(best_solution_pool, columns=knob_data['columnlabels'])
name = utils.get_filename('final_solutions', 'best_config', '.csv')
final_solution_pool.to_csv(os.path.join('final_solutions', name))
logger.info(f"save best config to {os.path.join('final_solutions', name)}")