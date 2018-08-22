# Copyright 2018 Daniel Hernandez Diaz, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected

from neurolib.models.model import Unsupervised
from neurolib.models.trainer import LLTrainer
from neurolib.utils.graphs import get_session


class DeterministicAE1D(Unsupervised):
  """
  """
  def __init__(self, Y, zdim, t_init=None, specs=None, trainer=LLTrainer, lr=1e-5):
    """
    TODO: Implement batch size
    """
    super().__init__(Y=Y)
    
    self.zdim = zdim
    with tf.variable_scope('DetAE', reuse=tf.AUTO_REUSE):
      if t_init is not None:
        self._initialize_from_tensors(t_init)
      else:
        if specs is not None:
          self._initialize_from_specs(specs)
        else:
          self.Yprime, self.Z = self._define_default()
      
      self.trainer = trainer = trainer(lr)
      self.train_op, self.loss, self.grads = trainer.build_cost_grads(Y, self.Yprime)
      
      self.Yfake = self._make_fake()
        
  def _generate_out(self, Z):
    """
    """
    ydim  = self.ydim
    full_out_1 = fully_connected(Z, 64)
    full_out_2 = fully_connected(full_out_1, 128)
    Yprime = fully_connected(full_out_2, ydim)
    
    return Yprime
  
  def _define_default(self):
    """
    """
    Y = self.Y
    zdim = self.zdim
    
    full_in_1 = fully_connected(Y, 128)
    full_in_2 = fully_connected(full_in_1, 64)
    Z = fully_connected(full_in_2, zdim)

    Yprime = self._generate_out(Z)    
    return Yprime, Z
  
  def _initialized_from_specs(self):
    """
    """
    pass
  
  def _initialize_from_tensors(self):
    """
    """
    
  def _make_fake(self, specs=None, num_samples=1000):
    """
    TODO: create a function for the variables initializer.
    TODO: num_samples has to be an argument of generate!
    """
    if specs is not None:
      pass
    else:
      Z = tf.get_variable('Z', initializer=np.random.randn(num_samples, self.zdim))
    return self._generate_out(Z)
      
  def generate(self):
    """
    """
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    return sess.run(self.Yfake)
      
  def update(self, ytrain):
    """
    TODO: construct the feed_dict
    """
    sess = get_session()
    _, loss = sess.run([self.train_op, self.loss], feed_dict={'Y:0' : ytrain})
    return loss
    
  def train(self, ytrain, num_epochs=5):
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    for _ in range(num_epochs):
      loss = self.update(ytrain)
      print(loss)
