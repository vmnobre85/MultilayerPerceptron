# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:22:17 2021

@author: Victor Nobre
"""

import numpy as np

#DETERMINANDO FUNCÕES DE ATIVAÇÃO.
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)