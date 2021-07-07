clear

best_configs_file_name=$1

python train.py
python parse_best_conf.py --n ./final_solutions/$best_configs_file_name
sshpass -p 1423 scp /home/jieun/RS-OtterTune/tuner/config0.cnf jieun@10.178.15.229:/home/jieun/data_generation_rocksdb/conf_tmp/