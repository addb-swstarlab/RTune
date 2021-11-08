Database systems typically have a large number of knobs that must be configured by database administrators and users to achieve high performance. RocksDB achieves fast data writing performance using a log-structured merge-tree. This database contains many knobs related to write and space amplification, which are important performance indicators in RocksDB. Write amplification degrades the database performance, whereas space amplification increases storage space owing to the storage of redundant data. Previously, it was proved that significant performance improvements could be achieved by tuning the database knobs. However, tuning multiple knobs simultaneously is a laborious task due to the large number of potential configuration combinations and trade-offs.
To address this problem, we built a system for RocksDB knob tuning. First, we generated a valuable RocksDB data repository for data analysis and tuning. The RocksDB data repository includes 20,000 random configurations for 16 different workloads, and the total number of pairs of data is 320,000. To find the workload that is most similar to a target workload, we created a new representation for workloads. We then applied the Mahalanobis distance to the data repository and a target workload to create a combined workload that is as close to the original target workload as possible and regarded the combined workload as the target workload. Subsequently, we trained a deep neural network model with the combined workload and used it as the fitness function of a genetic algorithm. Finally, we applied the genetic algorithm to find the best solution for the original target workload. The experimental results demonstrated that the proposed system achieved a significant performance improvement for various target workloads. Moreover, we conducted several experiments for performance comparison, including metrics pruning, knob pruning, workload combining, varying weights, and different hardware environments. 
We have generated 16 basic RocksDB workloads for tuning and analysis, and 6 target workloads for testing the system.
Using the above system, we successfully found the best solutions for different workloads and hardware environments.
We conducted the experiments among default settings, facebook recommendation, DBA recommendation and our model and avhieved exellent performance except one target workload.
Some study has mentioned that using pruned internal metrics may ahieve the similar performance when tuing a database, but the experiment carried by using pruned internal metrics showed that it is still better to use full internal metrcis for it contains more information about a workload. 
In addition, using only a little important konbs could also achieve great performance according to the previous study. Thus, we carry out the expriment using 3, 5, 7 important knobs for tuning, and the selection progress is done by random forest. The results showed that tuning all of the knobs at the same time achieved better performance in all target workloads.
We tried to adjust the weight of the score function to find the best external metrics respectively, but the model was hard to find the optimal configuration for the high-weighted external metrics.
To find the best solution is optimal for various hardware environments, we carried out the experiments of using the same optimal configuration for 4 different hardware environments,respectively. The results showed that the optimal configuration could achieve excellent performance among all of the target workloads.

