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

from neurolib.encoder.encoder import InnerNode
from neurolib.encoder import MultivariateNormalTriL  # @UnresolvedImport

act_fn_dict = {'relu' : tf.nn.relu,
               'leaky_relu' : tf.nn.leaky_relu}


class NormalTrilEncoding(InnerNode):
  """
  """
  def __init__(self, label, output_shapes, directives={}):
    """
    TODO: The user should be able to pass a tensorflow graph directly. In that
    case, EncoderNode should act as a simple wrapper that returns the input and the
    output.
    """    
    super(NormalTrilEncoding, self).__init__(label, output_shapes)
    self._update_directives(directives)

  def _update_directives(self, directives):
    """
    """
    self.directives = {'num_layers_0' : 2,
                      'num_nodes_0' : 128,
                      'activation_0' : 'leaky_relu',
                      'net_grow_rate_0' : 1.0,
                      'share_params' : False}
    self.directives.update(directives)
    
    # Deal with directives that should map to tensorflow objects hidden from the client
    self.directives['activation_0'] = act_fn_dict[self.directives['activation_0']]
    
  def _build(self, inputs=None):
    """
    Builds the graph corresponding to a single encoder.
    
    TODO: Expand this a lot, many more specs necessary.
    """
    dirs = self.directives
    if inputs is not None:
      raise NotImplementedError("") # TODO: Should I provide this option? meh
    if self.num_inputs > 1:
      # TODO. Implement merging of inputs, there may be different strategies, I
      # believe that for a first demonstration, simple concatenation suffices
#       merge_inputs(...)
      x_in = tf.concat(list(self.inputs.values()), axis=1)
    else:
      # This was set while the BFS was working on the parent node
      x_in = self.inputs[0]
    
    for oslot in range(self.num_outputs):
      num_layers = dirs['num_layers_' + str(oslot)]
      num_nodes = dirs['num_nodes_' + str(oslot)]
      activation = dirs['activation_' + str(oslot)]
      net_grow_rate = dirs['net_grow_rate_' + str(oslot)]

      # Define the Means
      output_dim = self.output_shapes[oslot][0]
      hid_layer = fully_connected(x_in, num_nodes, activation_fn=activation,
            biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
      for _ in range(num_layers-1):
        num_nodes = int(num_nodes*net_grow_rate)
        hid_layer = fully_connected(hid_layer, num_nodes, activation_fn=activation,
            biases_initializer=tf.random_normal_initializer(stddev=1/np.sqrt(num_nodes)))
      self.mean = fully_connected(hid_layer, output_dim, activation_fn=None)
      self.mean = tf.identity(self.mean, name="Mean_" + str(self.label) + '_' + str(oslot))
    
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
        output_chol = fully_connected(hid_layer, output_dim**2, activation_fn=None)
      output_chol = tf.reshape(output_chol, shape=[1, output_dim, output_dim])

      self.cholesky_tril = tf.matrix_band_part(output_chol, -1, 0,
                                        name='CholTril_' + str(self.label) + '_' + str(oslot))
      self.dist = MultivariateNormalTriL(loc=self.mean, scale_tril=self.cholesky_tril)
      self.outputs[oslot] = self.dist.sample(name='Out' + str(self.label) + '_' + str(oslot))

    self._is_built = True    
