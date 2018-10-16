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

from neurolib.encoder.basic import InnerNode
from neurolib.encoder import MultivariateNormalTriL  # @UnresolvedImport

act_fn_dict = {'relu' : tf.nn.relu,
               'leaky_relu' : tf.nn.leaky_relu}


class NormalTriLNode(InnerNode):
  """
  """
  num_expected_inputs = 1
  num_expected_outputs = 3
  
  def __init__(self, label, output_shape, builder, name=None,
               batch_size=1, directives={}):
    """
    TODO: The user should be able to pass a tensorflow graph directly. In that
    case, InnerNode should act as a simple wrapper that returns the input and the
    output.
    """
    self.name = "NormalTril_" + str(label) if name is None else name
    self.builder = builder
    self.num_inputs = 0
    self.batch_size = batch_size
    super(NormalTriLNode, self).__init__(label, output_shape)
    print(self.label)
    
    if isinstance(output_shape, int):
      main_oshape = [batch_size] + [output_shape]
    elif isinstance(output_shape, list):
      if isinstance(output_shape[0], int):
        main_oshape = [batch_size] + output_shape
      elif isinstance(output_shape[0], list):
        main_oshape = [batch_size] + output_shape[0]
    else:
      raise ValueError("The output_shape of a DeterministicNode must be an int or "
                       "a list of ints")
    self.main_oshape = self._oslot_to_shape[0] = main_oshape
#     self._oslot_to_shape[0] = main_oshape
    
    self._update_directives(directives)

    self._declare_secondary_outputs()
    
  def _declare_secondary_outputs(self):
    """
    """
    main_oshape = self._oslot_to_shape[0]
    
    # Mean oslot
    self._oslot_to_shape[1] = main_oshape
    o1 = self.builder.addOutput(name=self.directives['output_mean_name'])
    self.builder.addDirectedLink(self, o1, oslot=1)
    
    # Std oslot
    self._oslot_to_shape[2] = main_oshape + [main_oshape[-1]]  
    o2 = self.builder.addOutput(name=self.directives['output_cholesky_name'])
    
    print('_oslot_to_shape', self._oslot_to_shape)
    self.builder.addDirectedLink(self, o2, oslot=2)

  def _update_directives(self, directives):
    """
    """
    self.directives = {'num_layers' : 2,
                      'num_nodes' : 128,
                      'activation' : 'leaky_relu',
                      'net_grow_rate' : 1.0,
                      'share_params' : False,
                      'output_mean_name' : self.name + '_mean',
                      'output_cholesky_name' : self.name + '_cholesky'}
    self.directives.update(directives)
    
    # Deal with directives that map to tensorflow objects hidden from the client
    self.directives['activation'] = act_fn_dict[self.directives['activation']]

  @InnerNode.num_inputs.setter
  def num_inputs(self, value):
    """
    Sets self.num_inputs
    """
    if value > self.num_expected_inputs:
      raise ValueError("Attribute num_inputs of DeterministicNode must "
                           "should not be greather than ", self.num_expected_inputs)
    self._num_declared_inputs = value

  @InnerNode.num_outputs.setter
  def num_outputs(self, value):
    """
    Sets self.num_outputs
    """
    if value > self.num_expected_outputs:
      raise AttributeError("Attribute num_outputs of NormalTriLNode must "
                           "should not be greather than ", self.num_expected_outputs)
    self._num_declared_outputs = value

  def _build(self, inputs=None):
    """
    Builds the graph corresponding to a NormalTriL encoder.
    
    TODO: Expand this a lot, many more specs necessary.
    """
    dirs = self.directives
    if inputs is not None:
      raise NotImplementedError("") # TODO: Should I provide this option? meh

    x_in = self._islot_to_itensor[0]
    
    num_layers = dirs['num_layers']
    num_nodes = dirs['num_nodes']
    activation = dirs['activation']
    net_grow_rate = dirs['net_grow_rate']

    # Define the Means
    output_dim = self._oslot_to_shape[0][-1] # Last dim
    hid_layer = fully_connected(x_in, num_nodes, activation_fn=activation,
          biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
    for _ in range(num_layers-1):
      num_nodes = int(num_nodes*net_grow_rate)
      hid_layer = fully_connected(hid_layer, num_nodes, activation_fn=activation,
          biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
    mean = fully_connected(hid_layer, output_dim, activation_fn=None)
    
    # Define the Cholesky Lower Decomposition
    if dirs['share_params']:
      output_chol = fully_connected(hid_layer, output_dim**2, activation_fn=None)
    else:
      hid_layer = fully_connected(x_in, num_nodes, activation_fn=activation,
          biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
      for _ in range(num_layers-1):
        num_nodes = int(num_nodes*net_grow_rate)
        hid_layer = fully_connected(hid_layer, num_nodes, activation_fn=activation,
          biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
      output_chol = fully_connected(hid_layer, output_dim**2,
          activation_fn=None,
          weights_initializer = tf.random_normal_initializer(stddev=1e-4),
#           normalizer_fn=lambda x : x/tf.sqrt(x**2),
          biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(output_dim**2)))
    output_chol = tf.reshape(output_chol, 
#                              shape=[self.batch_size, output_dim, output_dim])
                             shape=[-1, output_dim, output_dim])

    if 'output_mean_name' in self.directives:
      mean_name = self.directives['output_mean_name']
    else:
      mean_name = "Mean_" + str(self.label) + '_0'
    if 'output_cholesky_name' in self.directives:
      cholesky_name = self.directives['output_cholesky_name']
    else:
      cholesky_name = 'CholTril_' + str(self.label) + '_0'
    
    cholesky_tril = tf.identity(output_chol, name=cholesky_name)
    
    # Get the tensorflow distribution for this node
    self.dist = MultivariateNormalTriL(loc=mean, scale_tril=cholesky_tril)

    # Fill the oslots
    self._oslot_to_otensor[0] = self.dist.sample(name='Out' + 
                                                 str(self.label) + '_0')
    self._oslot_to_otensor[1] = tf.identity(mean, name=mean_name)
    self._oslot_to_otensor[2] = cholesky_tril
    
    self._is_built = True
    
  def _log_prob(self, ipt):
    """
    """    
    return self.dist.log_prob(ipt)


