#!/bin/bash

clear

beginTime=$(date +%s%N)

# initiate the comp_results.csv file
sshpass -p 1423 ssh jieun@10.178.15.229 "echo "index,TIME,RATE,WAF,SA" > comp_results.csv"

#for (( i=22; i<=22; i++ ))
for (( i=16; i<=21; i++ ))
#for (( i=16; i<=16; i++ ))
do
    #let cal_num=$i-22+${11}
    let cal_num=$i-16+${11}
    if [ $cal_num -lt 10 ]
    then
        cal_num="0$cal_num"
    fi
    best_configs_file_name="20211104/best_config-20211104-$cal_num.csv"
    # train model and get the best configuration
    python train.py --iscombined $2 --isskip $3 --topk $4 --balance $5 --balance $6 --balance $7 --balance $8 --mode "$9" --exmetric "${10}" --target $i 
    # parsing the best configuration with the form of .cnf
    python parse_best_conf.py --n ./final_solutions/$best_configs_file_name
    # send the best configuration to the test server (data-generation-28)   
    sshpass -p 1423 scp /home/jieun/RS-OtterTune/tuner/config0.cnf jieun@10.178.15.229:/home/jieun/data_generation_rocksdb/conf_tmp/
    # test the best configuration on the test server (data-generation-28)
    sshpass -p 1423 ssh jieun@10.178.15.229 ./testing.sh $i
done

sshpass -p 1423 ssh jieun@10.178.15.229 "cp comp_results.csv $1.csv"

endTime=$(date +%s%N)
elapsed=`echo "($endTime - $beginTime) / 1000000" | bc`
elapsedSec=`echo "scale=2;$elapsed / 1000" | bc | awk '{printf "%d", $1}'`
echo 
echo TOTAL: $elapsedSec sec
echo 

# python train.py --mode "multi" --exmetric "SCORE" --iscombined true --topk 5 --target 21