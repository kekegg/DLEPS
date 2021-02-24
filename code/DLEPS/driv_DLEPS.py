########################################################
# All rights reserved. 
# Author: XIE Zhengwei @ Beijing Gigaceuticals Tech Co., Ltd 
#                      @ Peking University International Cancer Institute
# Contact: xiezhengwei@gmail.com
#
#
########################################################

import pickle
import numpy as np
import time
import argparse
import os
from shutil import copyfile
import sys, os

from dleps_predictor import DLEPS

import pandas as pd

INPUTFILE = "input.csv"

def get_arguments():
    parser = argparse.ArgumentParser(description='DLEPS main script')
    parser.add_argument('--input', default=INPUTFILE,
                        help='Brief format of chemicals: contains SMILES column. ')
    parser.add_argument('--use_onehot',  default=True,
                        help='If use pre-stored one hot array to save time.')
    parser.add_argument('--use_l12k',  default=None,
                        help='Use pre-calculated L12k')
    parser.add_argument('--upset',  default=None,
                        help='Up set of genes')
    parser.add_argument('--downset',  default=None,
                        help='Down set of genes. ')
    parser.add_argument('--reverse',  default=True,
                        help='Reverse the Up / Down set of genes. ')
    parser.add_argument('--output',  default='out.csv',
                        help='Output file name. ')
    return parser.parse_args()

def main():
    args = get_arguments()

    
    if args.use_l12k:
        cs = cs_driv(args.use_l12k, up_name=args.upset, down_name=args.downset, smiles_fn = args.input)
        cs.to_csv(args.output)
    else: 
        predictor = DLEPS(reverse=args.reverse, up_name=args.upset, down_name=args.downset)

        onehot_file = args.input+'_onehot'
        dt = pd.read_csv(args.input)

        if os.path.isfile(onehot_file+'.npy'):
            print("Using saved one hot file. \n")
            scores = predictor.predict(dt['SMILES'].values, load_onehot=onehot_file+'.npy')
        elif os.path.isfile(onehot_file+'.npz'):
            print("Using saved one hot file. \n")
            scores = predictor.predict(dt['SMILES'].values, load_onehot=onehot_file+'.npz')
        else:
            scores = predictor.predict(dt['SMILES'].values, save_onehot=onehot_file)

        dt['score'] = scores
        dt.to_csv(args.output)
        print("Output has been saved at : " + args.output)

if __name__ == '__main__':
    main()    
