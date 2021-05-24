#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:40:03 2021

@author: gombosurenatarbayr
"""



# 2D Poisson;


# import libs;

from __future__ import absolute_import;
from __future__ import division;
from __future__ import print_function;

import matplotlib.pyplot as plt;
import numpy as np;

import tensorflow as tf;
import deepxde as dde;
from deepxde.backend import tf;
#%matplotlib qt




def pde (x, y):
    dy_x = tf.gradients(y, x)[0];
    dy_x, dy_y = dy_x[:, 0:1], dy_x[:, 1:];
    dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]; 
    dy_yy = tf.gradients(dy_y, x)[0][:, 1:];
    return - dy_xx - dy_yy - 1 - x*y;


def boundary (x, on_boundary):
    return on_boundary;


def func (x):
    return np.zeros ([len(x) , 1]);


geom = dde.geometry.Polygon([
    [0,0],
    [1,0],
    [1,-1],
    [-1,-1],
    [-1,1],
    [0,1]
    ]);
bc = dde.DirichletBC(geom, func, boundary);
data = dde.data.PDE(geom, pde, bc, num_domain = 1000, num_boundary = 100, num_test = 1500);


net = dde.maps.FNN([2] + [50]*4 + [1] , "tanh", "Glorot uniform");


model = dde.Model(data, net);
model.compile ("adam", lr = 0.001);


checkpointer = dde.callbacks.ModelCheckpoint(
    "/Users/gombosurenatarbayr/Desktop/DL_for_PDE_model/checkpoints_poisson2D/model_poisson2d.ckpt", verbose = 1, save_better_only = True
)

losshistory, train_state = model.train(
    epochs = 5000, callbacks=[checkpointer]
);


dde.saveplot(losshistory, train_state, issave = True, isplot = True);









