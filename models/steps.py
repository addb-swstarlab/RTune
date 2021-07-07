import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader
import torch, sys, random

from models.cluster import KMeansClusters, create_kselection_model
from models.factor_analysis import FactorAnalysis
from models.preprocessing import (get_shuffle_indices, consolidate_columnlabels)
from models.lasso import LassoPath
from models.xgboost import XGBR
from models.util import DataUtil
from models.rf import RFR
from models.gp import GPRNP
from models.parameters import *
from models.denseGA import RocksDBDataset, Net, train, valid, fitness_function
import models.rocksdb_option as option
sys.path.append('../')
from tuner.utils import *


device = torch.device("cpu")

## Try Scaling 
def run_workload_characterization(metric_data, cluster_threshold=20, skip=False):
    ##TODO: modift after workload generation.

    matrix = metric_data['data']
    columnlabels = metric_data['columnlabels']

    # # Bin each column (metric) in the matrix by its decile
    # binner = Bin(bin_start=1, axis=0)
    # binned_matrix = binner.fit_transform(matrix)

    # Remove any constant columns
    nonconst_matrix = []
    nonconst_columnlabels = []
    for col, (_,v) in zip(matrix.T, enumerate(columnlabels)):
        if np.any(col != col[0]):
            #print(col.reshape(-1, 1))
            nonconst_matrix.append(col.reshape(-1, 1))
            nonconst_columnlabels.append(v)
    assert len(nonconst_matrix) > 0, "Need more data to train the model"
    nonconst_matrix = np.hstack(nonconst_matrix)
    print("Workload characterization ~ nonconst data size: %s", nonconst_matrix.shape)

    # Remove any duplicate columns
    unique_matrix, unique_idxs = np.unique(nonconst_matrix, axis=1, return_index=True)
    unique_columnlabels = [nonconst_columnlabels[idx] for idx in unique_idxs]

    unique_matrix = nonconst_matrix
    unique_idxs = unique_idxs
    unique_columnlabels = nonconst_columnlabels

    print("Workload characterization ~ final data size: %s", unique_matrix.shape)
    n_rows, n_cols = unique_matrix.shape

    if skip:
        print("Skipping pruning Internal Metrics data")
        return nonconst_columnlabels

    # Shuffle the matrix rows
    shuffle_indices = get_shuffle_indices(n_rows)
    shuffled_matrix = unique_matrix[shuffle_indices, :]

    fa_model = FactorAnalysis()
    fa_model.fit(shuffled_matrix, unique_columnlabels, n_components=5)

    # Components: metrics * factors
    components = fa_model.components_.T.copy()

    kmeans_models = KMeansClusters()
    ##TODO: Check Those Options
    kmeans_models.fit(components, min_cluster=1,
                      max_cluster=min(n_cols - 1, cluster_threshold),
                      sample_labels=unique_columnlabels,
                      estimator_params={'n_init': 50})

    # Compute optimal # clusters, k, using gap statistics
    gapk = create_kselection_model("gap-statistic")
    gapk.fit(components, kmeans_models.cluster_map_)

    print("Found optimal number of clusters: {}".format(gapk.optimal_num_clusters_))

    # Get pruned metrics, cloest samples of each cluster center
    pruned_metrics = kmeans_models.cluster_map_[gapk.optimal_num_clusters_].get_closest_samples()

    return pruned_metrics


def run_knob_identification(knob_data,metric_data,mode, logger):
    # TODO: type filter for Redis, RocksDB 
    
    knob_matrix = knob_data['data']
    knob_columnlabels = knob_data['columnlabels']

    metric_matrix = metric_data['data']
    #metric_columnlabels = metric_data['columnlabels']

    encoded_knob_columnlabels = knob_columnlabels
    encoded_knob_matrix = knob_matrix

    # standardize values in each column to N(0, 1)
    standardizer = StandardScaler()
    standardized_knob_matrix = standardizer.fit_transform(encoded_knob_matrix)
    standardized_metric_matrix = standardizer.fit_transform(metric_matrix)

    # shuffle rows (note: same shuffle applied to both knob and metric matrices)
    shuffle_indices = get_shuffle_indices(standardized_knob_matrix.shape[0], seed=17)
    shuffled_knob_matrix = standardized_knob_matrix[shuffle_indices, :]
    shuffled_metric_matrix = standardized_metric_matrix[shuffle_indices, :]

    if mode == 'lasso':
    # run lasso algorithm
        lasso_model = LassoPath()
        lasso_model.fit(shuffled_knob_matrix, shuffled_metric_matrix, encoded_knob_columnlabels)        
        encoded_knobs = lasso_model.get_ranked_features()
    elif mode == "XGB":
        xgb_model = XGBR()
        xgb_model.fit(shuffled_knob_matrix, shuffled_metric_matrix,encoded_knob_columnlabels)
        encoded_knobs = xgb_model.get_ranked_knobs()
        feature_imp = xgb_model.get_ranked_importance()
        logger.info('feature importance')
        logger.info(feature_imp)
    elif mode == "RF":
        rf = RFR()
        rf.fit(shuffled_knob_matrix,shuffled_metric_matrix,encoded_knob_columnlabels)
        encoded_knobs = rf.get_ranked_features()
        feature_imp = rf.get_ranked_importance()
        logger.info('feature importance')
        logger.info(feature_imp)

    consolidated_knobs = consolidate_columnlabels(encoded_knobs)

    return consolidated_knobs

# def run_workload_mapping(knob_data, metric_data, target_knob, target_metric, params):
#     '''
#     Args:
#         knob_data: train knob data
#         metric_data: train metric data
#         target_knob: target knob data
#         target_metric: target metric data
#     '''
#     #knob_data["data"],knob_data["columnlabels"] = DataUtil.clean_knob_data(knob_data["data"],knob_data["columnlabels"])

#     ##TODO: Will change dict to something
#     X_matrix = np.array(knob_data["data"])
#     y_matrix = np.array(metric_data["data"])
#     #rowlabels to np.arange(X_matrix.shape[0])
#     rowlabels = np.array(knob_data["rowlabels"])
#     assert np.array_equal(rowlabels, metric_data["rowlabels"])

#     X_matrix, y_matrix, rowlabels = DataUtil.combine_duplicate_rows(
#             X_matrix, y_matrix, rowlabels)

#     # If we have multiple workloads and use them to train,
#     # Workload mapping should be called (not implemented yet) and afterward,
#     # Mapped workload will be stored in workload_data.
#     workload_data = {}
#     unique_workload = 'UNIQUE'
#     workload_data[unique_workload] = {
#             'X_matrix': X_matrix,
#             'y_matrix': y_matrix,
#             'rowlabels': rowlabels,
#     }

#     if len(workload_data) == 0:
#         # The background task that aggregates the data has not finished running yet
#         target_data.update(mapped_workload=None, scores=None)
#         print('%s: Result = %s\n', task_name, _task_result_tostring(target_data))
#         print('%s: Skipping workload mapping because no different workload is available.',task_name)
#         return target_data, algorithm

#     Xs = np.vstack([entry['X_matrix'] for entry in list(workload_data.values())])
#     ys = np.vstack([entry['y_matrix'] for entry in list(workload_data.values())])

#     # Scale the X & y values, then compute the deciles for each column in y
#     X_scaler = StandardScaler(copy=False)
#     X_scaler.fit(Xs)
#     y_scaler = StandardScaler(copy=False)
#     y_scaler.fit_transform(ys)
#     y_binner = Bin(bin_start=1, axis=0)
#     y_binner.fit(ys)
#     del Xs
#     del ys

#     X_target = target_data['X_matrix']
#     # Filter the target's y data by the pruned metrics.
#     y_target = target_data['y_matrix'][:, pruned_metric_idxs]

#     # Now standardize the target's data and bin it by the deciles we just
#     # calculated
#     X_target = X_scaler.transform(X_target)
#     y_target = y_scaler.transform(y_target)
#     y_target = y_binner.transform(y_target)

#     predictions = np.empty_like(y_target)
#     X_workload = workload_data['X_matrix']
#     X_scaled = X_scaler.transform(X_workload)
#     y_workload = workload_data['y_matrix']
#     y_scaled = y_scaler.transform(y_workload)
#     for j, y_col in enumerate(y_scaled.T):
#         y_col = y_col.reshape(-1, 1)
#         model = GPRNP(length_scale=params['GPR_LENGTH_SCALE'],
#                         magnitude=params['GPR_MAGNITUDE'],
#                         max_train_size=params['GPR_MAX_TRAIN_SIZE'],
#                         batch_size=params['GPR_BATCH_SIZE'])
#         model.fit(X_scaled, y_col, ridge=params['GPR_RIDGE'])
#         gpr_result = model.predict(X_target)
#         predictions[:, j] = gpr_result.ypreds.ravel()
#     # Bin each of the predicted metric columns by deciles and then
#     # compute the score (i.e., distance) between the target workload and each of the known workloads
#     predictions = y_binner.transform(predictions)
#     dists = np.sqrt(np.sum(np.square(
#                 np.subtract(predictions, y_target)), axis=1))
#     scores[workload_id] = np.mean(dists)

#     # TODO: return minimum dist workload


# def configuration_recommendation(target_knob, target_metric, logger, gp_type='numpy', db_type='redis', data_type='RDB'):
#     X_columnlabels, X_scaler, X_scaled, y_scaled, X_max, X_min, _ = utils.process_training_data(target_knob, target_metric, db_type, data_type)

#     num_samples = params["NUM_SAMPLES"]
#     X_samples = np.empty((num_samples, X_scaled.shape[1]))
#     for i in range(X_scaled.shape[1]):
#         X_samples[:, i] = np.random.rand(num_samples) * (X_max[i] - X_min[i]) + X_min[i]

#     # q = queue.PriorityQueue()
#     # for x in range(0, y_scaled.shape[0]):
#     #     q.put((y_scaled[x][0], x))

#     # ## TODO : What...?
#     # i = 0
#     # while i < params['TOP_NUM_CONFIG']:
#     #     try:
#     #         item = q.get_nowait()
#     #         # Tensorflow get broken if we use the training data points as
#     #         # starting points for GPRGD. We add a small bias for the
#     #         # starting points. GPR_EPS default value is 0.001
#     #         # if the starting point is X_max, we minus a small bias to
#     #         # make sure it is within the range.
#     #         dist = sum(np.square(X_max - X_scaled[item[1]]))
#     #         if dist < 0.001:
#     #             X_samples = np.vstack((X_samples, X_scaled[item[1]] - abs(params['GPR_EPS'])))
#     #         else:
#     #             X_samples = np.vstack((X_samples, X_scaled[item[1]] + abs(params['GPR_EPS'])))
#     #         i = i + 1
#     #     except queue.Empty:
#     #         break
#     res = None
#     if gp_type == 'numpy':
#         # DO GPRNP
#         model = GPRNP(length_scale = params["GPR_LENGTH_SCALE"],
#                         magnitude=params["GPR_MAGNITUDE"],
#                         max_train_size=params['GPR_MAX_TRAIN_SIZE'],
#                         batch_size=params['GPR_BATCH_SIZE'])
#         model.fit(X_scaled,y_scaled,ridge=params["GPR_RIDGE"])
#         res = model.predict(X_samples).ypreds
#         logger.info('do GPRNP')
#         del model
#     elif gp_type == 'scikit':
#         # # DO SCIKIT-LEARN GP
#         # model = GaussianProcessRegressor().fit(X_scaled,y_scaled)
#         # res = model.predict(X_samples)
#         # print('do scikit-learn gp')

#         from sklearn.gaussian_process.kernels import DotProduct
#         GPRkernel = DotProduct(sigma_0=0.5)
#         model = GaussianProcessRegressor(kernel = GPRkernel,
#                             alpha = params["ALPHA"]).fit(X_scaled,y_scaled)
#         res = model.predict(X_samples)
#         del model
#     else:
#         raise Exception("gp_type should be one of (numpy and scikit)")

#     best_config_idx = np.argmax(res.ravel())
#     if len(set(res.ravel()))==1:
#         logger.info("FAIL TRAIN")
#         return False, -float('inf'), None
#     best_config = X_samples[best_config_idx, :]
#     best_config = X_scaler.inverse_transform(best_config)
#     X_min_inv = X_scaler.inverse_transform(X_min)
#     X_max_inv = X_scaler.inverse_transform(X_max)
#     best_config = np.minimum(best_config, X_max_inv)
#     best_config = np.maximum(best_config, X_min_inv)
#     conf_map = {k: best_config[i] for i, k in enumerate(X_columnlabels)}
#     # logger.info("\n\n\n")
#     logger.info(conf_map)
#     #convert_dict_to_conf(conf_map, data_type)

#     logger.info("FINISH TRAIN")
#     print(np.max(res.ravel()))
#     return True, np.max(res.ravel()), conf_map

def get_ranked_knob_data(ranked_knobs, knob_data, top_k):
    '''
        ranked_knobs: sorted knobs with ranking 
                        ex. ['m3', 'm6', 'm2', ...]
        knob_data: dictionary data with keys(columnlabels, rowlabels, data)
        top_k: A standard to split knobs 
    '''
    ranked_knob_data = knob_data.copy()
    ranked_knob_data['columnlabels'] = np.array(ranked_knobs)
        
    for i, knob in enumerate(ranked_knobs):
        ranked_knob_data['data'][:,i] = knob_data['data'][:, list(knob_data['columnlabels']).index(knob)]
    
    # pruning with top_k
    ranked_knob_data['data'] = ranked_knob_data['data'][:,:top_k]
    ranked_knob_data['columnlabels'] = ranked_knob_data['columnlabels'][:top_k]

    #print('pruning data with ranking')
    #print('Pruned Ranked knobs: ', ranked_knob_data['columnlabels'])

    return ranked_knob_data

from scipy.spatial import distance

def numpy_to_df(numpy_):
    return pd.DataFrame(numpy_)

def generation_combined_workload(wk_internal_metrics, wk_external_metrics, target_wk, target_size, logger):
    '''
        wk_internal_metrics
        target_wk : target workload number
        target_size : target workload data size
    '''
    ## Extract statistical values
    wk_stats_list = []
    drop_columns = ['count', 'min', 'max']

    for i, wk_internal_metric in enumerate(wk_internal_metrics):
        df_wk_internal_metric = numpy_to_df(wk_internal_metric['data'])
        if i == target_wk:
            df_wk_internal_metric = df_wk_internal_metric.sample(target_size)
        wk_stats = df_wk_internal_metric.describe().T.drop(columns=drop_columns)
        wk_stats_list.append(wk_stats.T)

    ## Get Convariance from internal metric data
    cov_internal_metrics = pd.concat(wk_stats_list, axis=1).T.cov()

    ## Get Mahalanobis distance list
    int_idx = wk_stats_list[0].columns
    wk_mah_dis = {}
    for wk, wk_stats in enumerate(wk_stats_list):
        sum_d = 0
        for idx in int_idx:
            d = distance.mahalanobis(u=wk_stats[idx], v=wk_stats_list[target_wk][idx], VI=np.linalg.pinv(cov_internal_metrics))
            sum_d += d
        wk_mah_dis[wk] = sum_d
        logger.info(f"{wk}th workload get distance {sum_d}")

    # ## top sort - not using
    # top_sort = np.argsort(wk_mah_dis)[1:5]
    # top_score = np.sort(wk_mah_dis)[1:5]
    # sub_max_mah = max(top_score) - top_score

    ## Get workload score by Mahalanobis distance ranking
    sub_max_mah = {}
    reverse_score = {}
    for key in wk_mah_dis.keys():
        sub_max_mah[key] = max(wk_mah_dis.values()) - wk_mah_dis[key]
    sub_max_mah.pop(target_wk) # remove target wk

    for key in sub_max_mah.keys():
        reverse_score[key] = sub_max_mah[key]/sum(sub_max_mah.values())
    wk_sorted_score = sorted(reverse_score.items(), key=(lambda x:x[1]), reverse=True) # return [(),(),()] list, tuple
    
    logger.info(f"workload sorting by score : {wk_sorted_score}")
    # sub_max_mah = max(wk_mah_dis) - wk_mah_dis
    # reverse_score = sub_max_mah/sum(sub_max_mah)
    
    # wk_sorted_score = np.argsort(-reverse_score)[1:]
    


    ## Get combined workload by workload score
    wk_df_external_metrics = []
    for wk_external_metric in wk_external_metrics:
        df_external_metric = pd.DataFrame(data=wk_external_metric['data'], columns=wk_external_metric['columnlabels'], index=wk_external_metric['rowlabels'])
        wk_df_external_metrics.append(df_external_metric)

    sample_size = len(wk_df_external_metrics[0]) 
    # wk_sample_size = int(sample_size/len(wk_score))
    # if wk_sample_size*len(wk_score) != sample_size: # if there is a remainder after division
    #     wk_sample_size = wk_sample_size + (sample_size- wk_sample_size*len(wk_score))
    combined_wk = pd.DataFrame()

    ## whole sort
    # for wk in wk_sorted_score:
    #     wk_sample_size = round(reverse_score[wk] * sample_size)
    #     # Sorting external metrics by workload
    #     sorted_df_external_metric = wk_df_external_metrics[wk].sort_values(by=['WAF'])
    #     # Remove duplicated index
    #     sorted_df_external_metric = sorted_df_external_metric.drop(combined_wk.index)
    #     logger.info(f"{wk}th workload remained data size {len(sorted_df_external_metric)}, get data size {wk_sample_size}")
    #     combined_wk = pd.concat([combined_wk,sorted_df_external_metric[:wk_sample_size]])
    
    for _, (wk, score) in enumerate(wk_sorted_score):
        wk_sample_size = round(score * sample_size)
        # Sorting external metrics by workload
        sorted_df_external_metric = wk_df_external_metrics[wk].sort_values(by=['WAF'])
        # Remove duplicated index
        sorted_df_external_metric = sorted_df_external_metric.drop(combined_wk.index)
        logger.info(f"{wk}th workload remained data size {len(sorted_df_external_metric)}, get data size {wk_sample_size}")
        
        combined_wk = pd.concat([combined_wk,sorted_df_external_metric[:wk_sample_size]])

    if len(combined_wk) != sample_size:
        combined_wk = pd.concat([combined_wk,sorted_df_external_metric])

    # ## top sort
    # for i, wk in enumerate(top_sort):
    #     wk_sample_size = round(reverse_score[i] * sample_size)
    #     # Sorting external metrics by workload
    #     sorted_df_external_metric = wk_df_external_metrics[wk].sort_values(by=['WAF'])
    #     # Remove duplicated index
    #     sorted_df_external_metric = sorted_df_external_metric.drop(combined_wk.index)
    #     print(f"{wk}th workload remained data size {len(sorted_df_external_metric)}, get data size {wk_sample_size}")
        
    #     combined_wk = pd.concat([combined_wk,sorted_df_external_metric[:wk_sample_size]])
    return combined_wk.sort_index()

def configuration_recommendation(knob_data, combined_wk, logger, batch_size=64, epochs=300, lr=0.0001, n_pool=32, n_generation=10000):
    configs = knob_data['data']
    X_tr, X_te, y_tr, y_te = train_test_split(configs, np.array(combined_wk), test_size=0.2, random_state=42, shuffle=True)
    logger.info(f"X_train : {X_tr.shape} X_test : {X_te.shape} Y_train : {y_tr.shape} Y_test : {y_te.shape}")
    
    scaler_X = MinMaxScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(y_tr)

    X_norm_tr = scaler_X.transform(X_tr).astype(np.float32)
    X_norm_te = scaler_X.transform(X_te).astype(np.float32)
    y_norm_tr = scaler_y.transform(y_tr).astype(np.float32)
    y_norm_te = scaler_y.transform(y_te).astype(np.float32)

    Dataset_tr = RocksDBDataset(X_norm_tr, y_norm_tr)
    Dataset_te = RocksDBDataset(X_norm_te, y_norm_te)

    loader_tr = DataLoader(dataset = Dataset_tr, batch_size = batch_size, shuffle=True)
    loader_te = DataLoader(dataset = Dataset_te, batch_size = batch_size, shuffle=True)

    model = Net().to(device)

    losses_tr = []
    losses_te = []
    for epoch in range(epochs):
        loss_tr = train(model, loader_tr, lr)
        loss_te = valid(model, loader_te)
        
        losses_tr.append(loss_tr)
        losses_te.append(loss_te)
        
        if epoch % 10 == 0:
            logger.info(f"[{epoch:02d}/{epochs}] loss_tr: {loss_tr:.4f}\tloss_te:{loss_te:.4f}")
    logger.info(f"[{epoch:02d}/{epochs}] loss_tr: {loss_tr:.4f}\tloss_te:{loss_te:.4f}")

    name = get_filename('model_save', 'model', '.pt')
    torch.save(model, os.path.join('model_save', name))
    logger.info(f"save model to {os.path.join('model_save', name)}")

    ## GA Algorithm
    n_configs = configs.shape[1] # get number of 22
    n_pool_half = int(n_pool/2) # hafl of a pool size
    mutation = int(n_configs * 0.4) # mutate percentage

    current_solution_pool = configs[:n_pool] # dataframe -> numpy

    for i in range(n_generation):
        ## data scaling
        scaled_pool = scaler_X.transform(current_solution_pool)
        ## fitenss calculation with real values (not scaled)
        scaled = fitness_function(scaled_pool, model)
        fitness = scaler_y.inverse_transform(scaled)
        ## best parents selection
        idx_fitness = np.argsort(fitness)[:n_pool_half] # minimum
        best_solution_pool = current_solution_pool[idx_fitness,:]
        if i % 1000 == 999:
            logger.info(f"[{i+1:3d}/{n_generation:3d}] best fitness: {fitness[idx_fitness[0]]:.5f}")
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
    name = get_filename('final_solutions', 'best_config', '.csv')
    final_solution_pool.to_csv(os.path.join('final_solutions', name))
    logger.info(f"save best config to {os.path.join('final_solutions', name)}")