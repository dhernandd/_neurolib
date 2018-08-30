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

from neurolib.models.models import Unsupervised
from neurolib.models.trainer import MSETrainer
from neurolib.utils.graphs import get_session


class DeterministicAE1D(Unsupervised):
  """
  """
  def __init__(self, Y, zdim, t_init=None, specs=None, trainer=MSETrainer, lr=1e-5,
               method='adam'):
    """
    TODO: Implement batch size
    TODO: Document specs
    specs.keys() = ['in_nl', 'out_nl', 'num_in_layers', 'num_out_layers', 'in_nodes_list',
    'out_nodes_list']
    """
    self.zdim = zdim    
    super().__init__(Y=Y, t_init=t_init, specs=specs)

    if not self.is_initialized:
      self.Yprime, self.Z = self._build_default()
      
    self.trainer = trainer = trainer(lr=lr, method=method)
    self.train_op, self.loss, self.grads = trainer.build_cost_grads(Y, self.Yprime)
        
  def _build_out_default(self, Z):
    """
    """
    ydim  = self.ydim
#     with tf.variable_scope('DetAE', reuse=tf.AUTO_REUSE):
    full_out_1 = fully_connected(Z, 64)
    full_out_2 = fully_connected(full_out_1, 128)
    Yprime = fully_connected(full_out_2, ydim)
    return Yprime
  
  def _build_default(self):
    """
    """
    Y = self.Y
    zdim = self.zdim
    with tf.variable_scope('DetAE', reuse=tf.AUTO_REUSE):
      full_in_1 = fully_connected(Y, 128)
      full_in_2 = fully_connected(full_in_1, 64)
      Z = fully_connected(full_in_2, zdim)

      Yprime = self._build_out_default(Z)    
    return Yprime, Z
  
  def _initialized_from_specs(self):
    """
    """
    pass
  
  def _initialize_from_tensors(self):
    """
    TODO: Add the scope here to all variables
    """
    pass
  
  def _build_fake(self, t_init_gen=None, specs=None, nsamps=100):
    """
    TODO: create a function for the variables initializer.
    TODO: num_samples has to be an argument of generate!
    """
    if t_init_gen is not None:
      pass
    elif specs is not None:
      pass
    else:
      Z = tf.get_variable('Z', initializer=np.random.randn(nsamps, self.zdim))
      return self._build_out_default(Z)
      
  def _generate_out(self):
    """
    """
    pass
  
  def generate(self, t_init_gen=None, specs=None, nsamps=1000):
    """
    """
    Yfake = self._build_fake(t_init_gen, specs, nsamps=nsamps)
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    return sess.run(Yfake)
      
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


class BayesianAE1D(Unsupervised):
  """
  """
  def __init__(self, Y, zdim, t_init=None, specs=None, trainer=MSETrainer, lr=1e-5,
               method='adam'):
    """
    TODO: Implement batch size
    """
    self.zdim = zdim
    super().__init__(Y=Y, t_init=t_init, specs=specs)    
    if not self.is_initialized:
      self.Yprime, self.Z = self._define_default()
      
    self.trainer = trainer = trainer(lr=lr, method=method)
    self.train_op, self.loss, self.grads = trainer.build_cost_grads(Y, self.Yprime)
    
    self.Yfake = self._build_fake()
      
  
  
if __name__ == '__main__':
  Y = tf.placeholder(tf.float32, [1000,10], 'Y')
  model = DeterministicAE1D(Y, 2)
  ytrain = model.generate()
  print(ytrain[:5])