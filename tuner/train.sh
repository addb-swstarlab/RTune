clear

best_configs_file_name=$1

beginTime=$(date +%s%N)

python train.py --mode "multi" --exmetric "SCORE"
python parse_best_conf.py --n ./final_solutions/$best_configs_file_name
sshpass -p 1423 scp /home/jieun/RS-OtterTune/tuner/config0.cnf jieun@10.178.15.229:/home/jieun/data_generation_rocksdb/conf_tmp/

endTime=$(date +%s%N)
elapsed=`echo "($endTime - $beginTime) / 1000000" | bc`
elapsedSec=`echo "scale=2;$elapsed / 1000" | bc | awk '{printf "%d", $1}'`
echo 
echo TOTAL: $elapsedSec sec
echo 