# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:10:39 2023

@author: chongtm
"""


import sys 
sys.path.append("..") 
import numpy as np
from SWMM import SWMM_ENV
import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from swmm_api import read_out_file
import os


def HC_sample_action(foreaction,observation):
    action=[]
    k=0
    # CC R1
    if observation['CC-storage'] > 1.2:
        a = 1
    elif observation['CC-storage'] < 0.5:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # CC R2
    if observation['CC-storage'] > 1.4:
        a = 1
    elif observation['CC-storage'] < 0.5:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # CC S1
    if observation['CC-storage'] > 0.8:
        a = 1
    elif observation['CC-storage'] < 0.5:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # CC S2
    if observation['CC-storage'] > 1.0:
        a = 1
    elif observation['CC-storage'] < 0.5:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # JK R1
    if observation['JK-storage'] > 4.2:
        a = 1
    elif observation['JK-storage'] < 1.2:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # JK R2
    if observation['JK-storage'] > 4.3:
        a = 1
    elif observation['JK-storage'] < 1.2:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    k+=1
    # JK S
    if observation['JK-storage'] > 4.0:
        a = 1
    elif observation['JK-storage'] < 1.2:
        a = 0
    else:
        if foreaction[k] == 1:
            a = 1
        else:
            a = 0
    action.append(a)
    return action
