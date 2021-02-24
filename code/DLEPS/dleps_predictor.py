########################################################
# All rights reserved. 
# Author: XIE Zhengwei @ Beijing Gigaceuticals Tech Co., Ltd 
#                      @ Peking University International Cancer Institute
# Contact: xiezhengwei@gmail.com
#
#
########################################################

from __future__ import print_function
from __future__ import division

from os import path
import requests

import numpy as np
import pandas as pd
import nltk
import h5py
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from utils import get_fp, to1hot
import molecule_vae
from molecule_vae import get_zinc_tokenizer
import zinc_grammar
from vectorized_cmap import computecs

def sampling(args):
    z_mean_, z_log_var_ = args
    batch_size = K.shape(z_mean_)[0]
    epsilon = K.random_normal(shape=(batch_size, 56), mean=0., stddev = 0.01)
    return z_mean_ + K.exp(z_log_var_ / 2) * epsilon


class DLEPS(object):
    def __init__(self, setmean = False, reverse = True, base = -2, up_name=None, down_name=None, save_exp = None):

        
        self.save_exp = save_exp
        self.reverse = reverse
        self.base = base
        self.setmean = setmean
        self.loaded = False
        
        self.model = []
        self.model.append(self._build_model())
        self.W = self._get_W()
        A3, self.con = self._get_con()
        self.full_con = A3.dot(self.W)
        
        self.genes=pd.read_table("../../data/gene_info.txt",header=0)
        self.gene_dict=dict(list(zip(self.genes["pr_gene_symbol"], self.genes["pr_gene_id"])))
        
        if up_name:
            self.ups_new = self._get_genes(up_name)
        else:
            self.ups_new = None
            self.reverse = False
            print('DLEPS: No input of up files\n')

        if down_name:
            self.downs_new = self._get_genes(down_name)
        else:
            self.downs_new = None
            self.reverse = False
            print('DLEPS: No input of down files\n')

    def _build_model(self):
        # Variational autoencoder weights
        grammar_weights = '../../data/vae.hdf5'
        grammar_model = molecule_vae.ZincGrammarModel(grammar_weights)
        self.grammar_model = grammar_model
        z_mn, z_var = grammar_model.vae.encoderMV.output
        x = Lambda(sampling, output_shape=(56,), name='lambda')([z_mn, z_var])
        x = Dense(1024,activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(1024,activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(1024,activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(1024,activation='tanh')(x)
        x = Dropout(0.25)(x)
        expression = Dense(978,activation='linear')(x)
        model = Model(inputs = grammar_model.vae.encoderMV.input, outputs = expression)
        
        return model
    
    # Map the 978 genes to 12328 genes
    def _get_W(self):
        hf = h5py.File('../../data/denseweight.h5', 'r')
        n1 = hf.get('W')
        W = np.array(n1)
        return W
    
    # The average expression levels for 978 genes
    def _get_con(self):
        benchmark = pd.read_csv('../../data/benchmark.csv')
        A3 = np.concatenate((np.array([1]),benchmark['1.0'].values),axis=0)
        con = benchmark['1.0'].values
        return A3, con
    
    # Get the gene signatures
    def _get_genes(self, fl_name):
        print(fl_name)
        up = pd.read_csv(fl_name,header=None)
        ups= up.values.astype(int)
        print(ups.shape)
        ups = list(np.squeeze(ups))
        ups_new = [i for i in ups if i in list(self.genes["pr_gene_id"])]
        print(ups_new)
        return ups_new
    
    def train(self, smile_train, rna_train, validation_data, epochs=30000, batch_size=512, shuffle=True):
    
        assert (not self.loaded), 'Dense Model should not be loaded before training.'
        
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 10,
                                  min_lr = 0.0001)
        
        for layer in self.grammar_model.vae.encoderMV.layers:
            layer.trainable = False
        
        sgd = optimizers.SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
        rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        ada = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        adaD = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
        self.model[0].compile(optimizer='adadelta', loss='mean_squared_error')
        
        his = self.model[0].fit(smile_train, 
                rna_train, epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle, 
                validation_data=validation_data, # validation_data = (smile_val, rna_val)
                callbacks = [reduce_lr])
        
        return his
    
    # Only calculate the expression changes, clean smiles required
    def cal_expr(self, smiles, average=True):
        expr = []
        if len(smiles) > 0:
            onehot = to1hot(smiles)
            expr = self.model[0].predict(onehot)
            return expr
        else:
            return []

    # Compute the erichment score
    def comp_cs(self, expr, save_file):
        abs_expr = expr + self.con
        A2 = np.hstack([np.ones([expr.shape[0],1]), abs_expr])
        L12k = A2.dot(self.W)
        if self.setmean:
            L12k_df = pd.DataFrame(L12k, columns=self.genes["pr_gene_id"])
            L12k_df = L12k_df - L12k_df.mean()
        else:
            L12k_delta = L12k-self.full_con
            L12k_df = pd.DataFrame(L12k_delta, columns=self.genes["pr_gene_id"])
        cs = computecs(self.ups_new, self.downs_new, L12k_df.T)
        if save_file:
            L12k_df.T.to_hdf(save_file, key='data')
        return cs[0].values
        
    def predict(self, smiles, average=True, save_onehot=None, load_onehot=None, save_expr = None):
        # Dense network weights
        if not self.loaded:
            if path.exists('../../data/DLEPS_30000_tune_gvae10000.h5'):
                self.model[0].load_weights('../../data/DLEPS_30000_tune_gvae10000.h5')
                self.loaded = True
            
        score = []
        idx = []
        expr = []
        for i in range(len(smiles)):
            score.append(self.base)

        if load_onehot:
            clean_smiles = smiles
        else:
            fps = get_fp(smiles)
            assert len(smiles) == len(fps)
            clean_smiles = []
            clean_fps = []
            nan_smiles = []
            for i in range(len(fps)):
                if np.isnan(sum(fps[i])):
                    nan_smiles.append(smiles[i])
                else:
                    clean_smiles.append(smiles[i])
                    clean_fps.append(fps[i])
                    idx.append(i)
            clean_fps = np.array(clean_fps)

        if len(clean_smiles) > 0:
            if load_onehot:
                fss = load_onehot.split('.')
                if fss[-1] == 'npz':
                    onehotz = np.load(load_onehot)
                    print(onehotz.files)
                    onehot = onehotz[onehotz.files[0]]
                else:
                    onehot = np.load(load_onehot)
                head = '.'.join(fss[:-1])
                idx = np.load(head+'_idx.npy').astype(int)
            else:
                onehot = to1hot(clean_smiles)
                if save_onehot:
                    np.save(save_onehot,onehot)
                    np.save(save_onehot+'_idx',idx)
            if self.loaded:
                expr = self.model[0].predict(onehot)
            else:
                batch_s = 128
                if onehot.shape[0]>batch_s:
                    num_run = onehot.shape[0] // batch_s + 1
                    for ii in range(num_run):
                        if ii < num_run -1:
                            payload = {"instances": onehot[ii*batch_s:(ii+1)*batch_s].tolist()}
                        else:
                            payload = {"instances": onehot[ii*batch_s:].tolist()}
                        #For users in China
                        #res = requests.post("http://152.136.253.110:8018/v1/models/dleps:predict", json=payload)
                        #For users outside China
                        res = requests.post("http://161.35.239.224:8501/v1/models/dleps:predict", json=payload)
                        res = res.json()
                        if ii == 0:
                            expr = np.array(res['predictions'])
                        else:
                            expr = np.concatenate([expr, res['predictions']], axis = 0)
                else:
                    payload = {"instances": onehot.tolist()}
                    res = requests.post("http://161.35.239.224:8501/v1/models/dleps:predict", json=payload)
                    res = res.json()
                    expr = np.array(res['predictions'])
            print('DLEPS: 978 signatures obtained\n\n')
            step_size = 50000
            
            #If input SMILES is too large, split the files 
            if expr.shape[0] > step_size:
                cs_arr = np.zeros(expr.shape[0])
                num = int(expr.shape[0]/(step_size*1.0))
                for i in range(num):
                    print(i,num,sep=':')
                    cur_expr = expr[i*step_size:(i+1)*step_size]
                    if save_expr:
                        cur_cs = self.comp_cs(cur_expr, save_expr+'SMILES_L12k_'+str(i)+'.h5')
                    else:
                        cur_cs = self.comp_cs(cur_expr, None)
                    for j in range(cur_cs.shape[0]):
                        cs_arr[i*step_size+j] = cur_cs[j]

                cur_expr = expr[num*step_size:expr.shape[0]]
                if save_expr:
                    cur_cs = self.comp_cs(cur_expr, save_expr+'SMILES_L12k_'+str(num)+'.h5')
                else:
                    cur_cs = self.comp_cs(cur_expr,None)

                for j in range(cur_cs.shape[0]):
                    cs_arr[num*step_size+j] = cur_cs[j]
            else:
                abs_expr = expr + self.con
                A2 = np.hstack([np.ones([expr.shape[0],1]), abs_expr])
                L12k = A2.dot(self.W)
                if self.setmean:
                    L12k_df = pd.DataFrame(L12k, columns=self.genes["pr_gene_id"])
                    L12k_df = L12k_df - L12k_df.mean()
                else:
                    L12k_delta = L12k-self.full_con
                    L12k_df = pd.DataFrame(L12k_delta, columns=self.genes["pr_gene_id"])
                if self.save_exp:
                    L12k_df.to_csv(self.save_exp)

                print('DLEPS: 12328 gene changes obtained\n\n')
                if save_expr:
                    cs_arr = self.comp_cs(expr,save_expr)
                else:
                    cs = computecs(self.ups_new, self.downs_new, L12k_df.T)
                    cs_arr = cs[0].values
                print('DLEPS: Enrichment scores were calculated\n\n')

        for i in range(len(score)):
            if self.reverse:
                score[idx[i]] = cs_arr[i]*(-1)
            else:
                score[idx[i]] = cs_arr[i]

        return score
