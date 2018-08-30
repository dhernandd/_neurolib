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
import tensorflow as tf
from tensorflow.train import (AdamOptimizer, AdagradOptimizer, 
                              MomentumOptimizer, GradientDescentOptimizer)

class Trainer():
  """
  """
  opt_dict = {'adam' : AdamOptimizer, 'adagrad' : AdagradOptimizer,
              'momentum' : MomentumOptimizer, 'gd' : GradientDescentOptimizer}
  
  def __init__(self, lr, method):
    """
    """
    self.lr = lr
    self.optimizer = self.opt_dict[method]
    
  def build_cost_grads(self):
    """
    """
    raise NotImplementedError("")


class MSETrainer(Trainer):
  """
  """
  def __init__(self, lr, method='adam', **kwargs):
    """
    """
    super(MSETrainer, self).__init__(lr, method)
 
  
  def build_cost_grads(self, Y, Yprime):
    """
    """
    Nsamps = Y.get_shape().as_list()[0]
    cost = tf.reduce_sum(Y*Yprime)/Nsamps
    opt = self.optimizer(learning_rate=self.lr)
    self.train_step = tf.get_variable("global_step", [], tf.int64,
                                      tf.zeros_initializer(),
                                      trainable=False)

    self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=tf.get_variable_scope().name)
    print('Scope', tf.get_variable_scope().name)
    for i in range(len(self.train_vars)):
        shape = self.train_vars[i].get_shape().as_list()
        print("    ", i, self.train_vars[i].name, shape)

    gradsvars = opt.compute_gradients(cost, self.train_vars)
    train_op = opt.apply_gradients(gradsvars, global_step=self.train_step,
                                        name='train1_op')

    return train_op, cost, gradsvars
  

class LLTrainer(Trainer):
  """
  """
  def __init__(self, lr, method='adam', **kwargs):
    """
    """
    super(MSETrainer, self).__init__(lr, method)

  def build_cost_grads(self, Y, Yprime):
    """
    """