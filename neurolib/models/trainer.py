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

class Trainer():
  """
  """
  def __init__(self):
    """
    """
    pass


class LLTrainer(Trainer):
  """
  """
  def __init__(self, lr, **kwargs):
    """
    """
    self.lr = lr 
  
  def build_cost_grads(self, Y, Yprime):
    """
    """
    Nsamps = Y.get_shape().as_list()[0]
    cost = tf.reduce_sum(Y*Yprime)/Nsamps
    opt = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    self.train_step = tf.get_variable("global_step", [], tf.int64,
                                      tf.zeros_initializer(),
                                      trainable=False)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=tf.get_variable_scope().name)
    gradsvars = opt.compute_gradients(cost, train_vars)
    train_op = opt.apply_gradients(gradsvars, global_step=self.train_step,
                                        name='train1_op')

    return train_op, cost, gradsvars