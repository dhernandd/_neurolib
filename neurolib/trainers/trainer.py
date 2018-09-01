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

from neurolib.utils.graphs import get_session

class Trainer():
  """
  """
  opt_dict = {'adam' : tf.train.AdamOptimizer,
              'adagrad' : tf.train.AdagradOptimizer,
              'momentum' : tf.train.MomentumOptimizer,
              'gd' : tf.train.GradientDescentOptimizer}
  
  def __init__(self, train_specs):
    """
    """
    self.train_specs = train_specs
    
  def update(self):
    """
    """
    raise NotImplementedError("")

  def train(self):
    """
    """
    raise NotImplementedError("")


class GDTrainer(Trainer):
  """
  """
  def __init__(self, cost, train_specs, node_names):
    """
    """
    super(GDTrainer, self).__init__(train_specs)
    self.node_names = node_names
    self.cost = cost
    
    self.lr = lr = train_specs['lr']
    optimizer_class = self.opt_dict[train_specs['optimizer']]
    opt = optimizer_class(lr)
    self.train_step = tf.get_variable("global_step", [], tf.int64,
                                      tf.zeros_initializer(),
                                      trainable=False)

    self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=tf.get_variable_scope().name)

    gradsvars = opt.compute_gradients(cost, self.train_vars)
    self.train_op = opt.apply_gradients(gradsvars, global_step=self.train_step,
                                        name='train1_op')

  def update(self, ytrain):
    """
    TODO: construct the feed_dict
    """
    sess = get_session()
    _, cost = sess.run([self.train_op, self.cost],
                       feed_dict={self.node_names[0] : ytrain})
    return cost
  
  def train(self, ytrain, num_epochs=5, yvalid=None):
    """
    """
    nsamps = len(ytrain)
    sess = get_session()
    sess.run(tf.global_variables_initializer())
    for _ in range(num_epochs):
      for i in range(nsamps):
        loss = self.update(ytrain[i:i+1])
      print(loss)
      
    sess.close()
