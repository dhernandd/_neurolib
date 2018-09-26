import tensorflow as tf

from neurolib.trainers.costs import mse, elbo

cost_dict = {'mse' : mse,
             'elbo' : elbo}