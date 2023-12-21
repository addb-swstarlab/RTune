clear

# Get the best config file with the first input
best_configs_file_name=$1

python eval.py --model 20210709/model-20210709-04.pt --combined_wk 20210709/cbwk_0-20210709-04.csv

# Parse result for better visual understanding
python parse_best_conf.py --n ./final_solutions/$best_configs_file_name

# Copy the result file to the remote surver
sshpass -p 1423 scp /home/jieun/RS-OtterTune/tuner/config0.cnf jieun@10.178.15.229:/home/jieun/data_generation_rocksdb/conf_tmp/
