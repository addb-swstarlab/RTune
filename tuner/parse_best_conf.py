from configparser import ConfigParser
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=str, default='', help="best_configs file name")
opt = parser.parse_args()

def parse_best_conf(best_configs):
    bc = pd.read_csv(best_configs)
    bc = bc.iloc[: , 1:]
    bc = bc.iloc[0]
    temp_mem_ratio = bc.iloc[20]
    temp_comp_ratio = bc.iloc[21]
    bc = bc.astype(int)

    compression_type = ["snappy", "none", "lz4", "zlib"]
    bc.iloc[10] = compression_type[bc.iloc[10]]
    bc.iloc[20] = temp_mem_ratio
    bc.iloc[21] = temp_comp_ratio

    conf = ConfigParser()
    conf['rocksdb'] = bc
    with open('config0.cnf', 'w') as f:
        conf.write(f)

def main():
    parse_best_conf(opt.n)

if __name__ == '__main__':
    main()
