#!/bin/bash

# $1 is the file name
# python train.py --iscombined $2 --isskip $3 --topk $4 --balance $5 --mode "$6" --exmetric "$7"
# parser.add_argument("--iscombined", type=bool, default=True, help="Combine the workloads or not") 
# parser.add_argument("--isskip", type=bool, default=True, help="Skip the workload characterization or not") 
# parser.add_argument("--topk", type=int, default=-1, help="Define k to prune knob ranking data") 
# parser.add_argument("--balance", type=list, default=[0.25, 0.25, 0.25, 0.25], help="Define balance number to calculate score")
# parser.add_argument("--mode", type=str, default='dense', choices=['dense', 'multi'], help="Define which mode will use fitness function for GA in recommendation step")
# parser.add_argument("--exmetric", type=str, choices=["TIME", "RATE", "WAF", "SA", "SCORE"], default='RATE', help='Choose External Metrics')
# $8 is the index

# default command line
# ./train.sh file_name true true -1 [0.25, 0,25, 0.25, 0.25] multi SCORE 0

<< "END"
# test on 8.13
./train.sh "notcombined" false true -1 0.25 0.25 0.25 0.25 multi SCORE 0
./train.sh "notskip" true false -1 0.25 0.25 0.25 0.25 multi SCORE 6
./train.sh "topk" true true 3 0.25 0.25 0.25 0.25 multi SCORE 12
./train.sh "balance1144" true true -1 0.1 0.1 0.4 0.4 multi SCORE 18
./train.sh "balance4411" true true -1 0.4 0.4 0.1 0.1 multi SCORE 24
./train.sh "balance7111" true true -1 0.7 0.1 0.1 0.1 multi SCORE 30
./train.sh "balance1171" true true -1 0.1 0.1 0.7 0.1 multi SCORE 36
./train.sh "time" true true -1 0.25 0.25 0.25 0.25 dense TIME 42
./train.sh "waf" true true -1 0.25 0.25 0.25 0.25 dense WAF 48

# test on 8.17
./train.sh "notcombined" false true -1 0.25 0.25 0.25 0.25 multi SCORE 0
./train.sh "notskip" true false -1 0.25 0.25 0.25 0.25 multi SCORE 6
./train.sh "topk" true true 5 0.25 0.25 0.25 0.25 multi SCORE 12
./train.sh "balance1441" true true -1 0.1 0.4 0.4 0.1 multi SCORE 18
./train.sh "balance4114" true true -1 0.4 0.1 0.1 0.4 multi SCORE 24
./train.sh "balance1711" true true -1 0.1 0.7 0.1 0.1 multi SCORE 30
./train.sh "balance1117" true true -1 0.1 0.1 0.1 0.7 multi SCORE 36
./train.sh "rate" true true -1 0.25 0.25 0.25 0.25 dense RATE 42
./train.sh "saf" true true -1 0.25 0.25 0.25 0.25 dense SA 48


# test on 8.17
./train.sh "notcombined" false true -1 0.25 0.25 0.25 0.25 multi SCORE 0
./train.sh "notskip" true false -1 0.25 0.25 0.25 0.25 multi SCORE 6
./train.sh "topk" true true 7 0.25 0.25 0.25 0.25 multi SCORE 12
./train.sh "balance1441" true true -1 0.1 0.4 0.4 0.1 multi SCORE 18
./train.sh "balance4114" true true -1 0.4 0.1 0.1 0.4 multi SCORE 24
./train.sh "balance1144" true true -1 0.1 0.1 0.4 0.4 multi SCORE 30
./train.sh "balance4411" true true -1 0.4 0.4 0.1 0.1 multi SCORE 36
./train.sh "balance7111" true true -1 0.7 0.1 0.1 0.1 multi SCORE 42
./train.sh "balance1711" true true -1 0.1 0.7 0.1 0.1 multi SCORE 48
./train.sh "balance1171" true true -1 0.1 0.1 0.7 0.1 multi SCORE 54
./train.sh "balance1117" true true -1 0.1 0.1 0.1 0.7 multi SCORE 60
./train.sh "time" true true -1 0.25 0.25 0.25 0.25 dense TIME 66
./train.sh "rate" true true -1 0.25 0.25 0.25 0.25 dense RATE 72
./train.sh "waf" true true -1 0.25 0.25 0.25 0.25 dense WAF 78
./train.sh "saf" true true -1 0.25 0.25 0.25 0.25 dense SA 84

# test on 9.16
#./train.sh "balanceTIME" true true -1 1 0 0 0 multi SCORE 0
#./train.sh "balanceRATE" true true -1 0 1 0 0 multi SCORE 6
#./train.sh "balanceWAF" true true -1 0 0 1 0 multi SCORE 12
#./train.sh "balanceSA" true true -1 0 0 0 1 multi SCORE 24

# test on 9.29
#./train.sh "fillrandom" true true -1 0.25 0.25 0.25 0.25 multi SCORE 0
./train.sh "fillrandom" true true -1 0.25 0.25 0.25 0.25 multi SCORE 1
./train.sh "fillrandom" true true -1 0.25 0.25 0.25 0.25 multi SCORE 2
./train.sh "fillrandom" true true -1 0.25 0.25 0.25 0.25 multi SCORE 3

# test on 11.4
./train.sh "cos_1104" true true -1 0.25 0.25 0.25 0.25 multi SCORE 0
END

# test on 11.8
#./train.sh "cos_1108_1" true true -1 0.25 0.25 0.25 0.25 multi SCORE 0
#./train.sh "cos_1108_2" true true -1 0.25 0.25 0.25 0.25 multi SCORE 6
#./train.sh "cos_1108_3" true true -1 0.25 0.25 0.25 0.25 multi SCORE 12
./train.sh "cos_1108_4" true true -1 0.25 0.25 0.25 0.25 multi SCORE 18
./train.sh "cos_1108_5" true true -1 0.25 0.25 0.25 0.25 multi SCORE 24
./train.sh "cos_1108_6" true true -1 0.25 0.25 0.25 0.25 multi SCORE 30
./train.sh "cos_1108_7" true true -1 0.25 0.25 0.25 0.25 multi SCORE 36
./train.sh "cos_1108_8" true true -1 0.25 0.25 0.25 0.25 multi SCORE 42
