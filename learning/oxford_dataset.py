#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:45:16 2018
@author: landrieuloic
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg
import random

def get_datasets(args, test_seed_offset=0):
    """ Gets training and test datasets. """
    train_folders = ["2014-06-24-14-20-41", "2014-05-06-13-14-58"]
    test_folders = ["2014-05-14-13-59-05"]

    # Load superpoints graphs
    testlist, trainlist, validlist = [], [], []
    for tr in train_folders:
        path = '{}/superpoint_graphs/{}/'.format(args.OXFORD_PATH, tr)
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".h5"):
                if args.use_val_set:
                    flip_coin = random.random()
                    if flip_coin < 0.9:
                        #training set
                        trainlist.append(spg.spg_reader(args, path + fname, True))
                    else:
                        #validation set
                        validlist.append(spg.spg_reader(args, path + fname, True))
                else:
                    #training set
                    trainlist.append(spg.spg_reader(args, path + fname, True))


    #evaluation set
    for ts in test_folders:
        path = '{}/features_supervision/{}/'.format(args.ROOT_PATH, ts)
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".h5"):
                testlist.append(spg.spg_reader(args, path + fname, True))
  
    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist, validlist, scaler = spg.scaler01(trainlist, testlist, validlist=validlist)
        
    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.OXFORD_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.OXFORD_PATH, test_seed_offset=test_seed_offset)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in validlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.OXFORD_PATH, test_seed_offset=test_seed_offset)), \
            scaler


def get_info(args):

    semantic_colors_dict = {
        # 'void'          : [  0,  0,  0],
        'road'          : [128, 64,128],
        'sidewalk'      : [244, 35,232],
        'building'      : [ 70, 70, 70],
        'wall'          : [102,102,156],
        'fence'         : [190,153,153],
        'pole'          : [153,153,153],
        'traffic light' : [250,170, 30],
        'traffic sign'  : [220,220,  0],
        'vegetation'    : [107,142, 35],
        'terrain'       : [152,251,152],
        'sky'           : [ 70,130,180],
        'person'        : [220, 20, 60],
        'rider'         : [255,  0,  0],
        'car'           : [  0,  0,142],
        'truck'         : [  0,  0, 70],
        'bus'           : [  0, 60,100],
        'train'         : [  0, 80,100],
        'motorcycle'    : [  0,  0,230],
        'bicycle'       : [119, 11, 32],
        # 'outside camera': [255, 255, 0],
        # 'egocar'        : [123, 88,  4],
        #'unlabelled'    : [ 81,  0, 81]
	}
    
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1
    if args.loss_weights == 'none':
        weights = np.ones((20,),dtype='f4')
    else:
        weights = h5py.File(args.OXFORD_PATH + "/parsed/class_count.h5")["class_count"][:].astype('f4')
        weights = weights[:,[i for i in range(6) if i != args.cvfold-1]].sum(1)
        weights = (weights+1).mean()/(weights+1)
    if args.loss_weights == 'sqrt':
        weights = np.sqrt(weights)
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)
    return {
        'node_feats': 9 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'classes': 19,
        'class_weights': weights,
        'inv_class_map': {i:key for i,key in enumerate(semantic_colors_dict.keys())}
    }

def preprocess_pointclouds(OXFORD_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""
    class_count = np.zeros((20,6),dtype='int')
    folders = ["2014-06-24-14-20-41", "2014-05-06-13-14-58", "2014-05-14-13-59-05"]

    for folder in folders:
        pathP = '{}/parsed/{}/'.format(OXFORD_PATH, folder)
        pathD = '{}/features_supervision/{}/'.format(OXFORD_PATH, folder)
        pathC = '{}/superpoint_graphs/{}/'.format(OXFORD_PATH, folder)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(10)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                rgb = f['rgb'][:].astype(np.float)
                
                labels = f['labels'][:]
                hard_labels = np.argmax(labels[:,1:],1)
                label_count = np.bincount(hard_labels, minlength=13)
                class_count[:,n-1] = class_count[:,n-1] + label_count
                
                e = (f['xyz'][:,2][:] -  np.min(f['xyz'][:,2]))/ (np.max(f['xyz'][:,2]) -  np.min(f['xyz'][:,2]))-0.5

                rgb = rgb/255.0 - 0.5
                
                xyzn = (xyz - np.array([30,0,0])) / np.array([30,5,3])
                
                lpsv = np.zeros((e.shape[0],4))

                P = np.concatenate([xyz, rgb, e[:,np.newaxis], lpsv, xyzn], axis=1)

                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    hf.create_dataset(name='centroid',data=xyz.mean(0))
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])
    path = '{}/parsed/'.format(OXFORD_PATH)
    data_file = h5py.File(path+'class_count.h5', 'w')
    data_file.create_dataset('class_count', data=class_count, dtype='int')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--OXFORD_PATH', default='datasets/s3dis')
    args = parser.parse_args()
    preprocess_pointclouds(args.OXFORD_PATH)