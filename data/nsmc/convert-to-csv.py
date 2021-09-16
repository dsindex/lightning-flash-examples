import sys
import os
import argparse
import json
import pdb
import logging

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def proc(input_path):
    print("review,sentiment")
    tot_num_line = sum(1 for _ in open(input_path, 'r')) 
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            toks = line.strip().split('\t')
            if len(toks) <= 1: continue
            sent = toks[0].replace("\"", "\'")
            label = toks[1]
            if label == "1": label = "positive"
            if label == "0": label = "negative"
            print("\"{}\",{}".format(sent, label))

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', type=str, default='data/clova_sentiments/train.txt')
    opt = parser.parse_args()

    proc(opt.input_path)


if __name__ == '__main__':
    main()
