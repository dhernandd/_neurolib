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
from tensorflow.contrib.layers.python.layers import fully_connected

from neurolib.encoder.basic import EncoderNode

act_fn_dict = {'relu' : tf.nn.relu,
               'leaky_relu' : tf.nn.leaky_relu}


class DeterministicNode(EncoderNode):
  """
  """
  num_expected_inputs = 1
  num_expected_outputs = 1
  
  def __init__(self, label, output_shape, builder=None, batch_size=1,
               name=None, directives={}):
    """
    TODO: The user should be able to pass a tensorflow graph directly. In that
    case, EncoderNode should act as a simple wrapper that returns the input and the
    output.
    """
    self.name = "Det_" + str(label) if name is None else name
    self.builder = builder
    super(DeterministicNode, self).__init__(label,
                                                main_output_shapes=output_shape)
    
    self.num_outputs = 1
    if isinstance(output_shape, int):
      output_shape = [batch_size] + [output_shape]
    elif isinstance(output_shape, list):
      if isinstance(output_shape[0], int):
        output_shape = [batch_size] + output_shape
      elif isinstance(output_shape[0], list):
        output_shape = [batch_size] + output_shape[0]
    else:
      raise ValueError("The output_shape of a DeterministicNode must be an int or "
                       "a list of ints")
    self._oslot_to_shape[0] = output_shape
    
    self._update_directives(directives)

  @EncoderNode.num_inputs.setter
  def num_inputs(self, value):
    """
    Sets self.num_inputs
    """
    if value > self.num_expected_inputs:
      raise AttributeError("Attribute num_inputs of DeterministicNode must "
                           "should not be greather than ", self.num_expected_inputs)
    self._num_declared_inputs = value

  @EncoderNode.num_outputs.setter
  def num_outputs(self, value):
    """
    Sets self.num_outputs
    """
    if value > self.num_expected_outputs:
      raise AttributeError("Attribute num_outputs of DeterministicNode must "
                           "should not be greather than ", self.num_expected_outputs)
    self._num_declared_outputs = value

  def _update_directives(self, directives):
    """
    """
    self.directives = {'num_layers_0' : 2,
                       'num_nodes_0' : 128,
                       'activation_0' : 'relu',
                       'net_grow_rate_0' : 1.0}
    self.directives.update(directives)
    
    # Deal with directives that map to hidden tf objects.
    self.directives['activation_0'] = act_fn_dict[self.directives['activation_0']]
        
  def _build(self):
    """
    Builds the graph corresponding to a single encoder.
    """
    dirs = self.directives
    
    # At the moment, a Deterministic Node accepts only one input
    x_in = self._islot_to_itensor[0]

    num_layers = dirs['num_layers_0']
    num_nodes = dirs['num_nodes_0']
    activation = dirs['activation_0']
    net_grow_rate = dirs['net_grow_rate_0']
  
    print("self.num_outputs", self.num_outputs)
    # Build the neural network from dirs
    for oslot in range(self.num_outputs):
      output_name = self.name + '_out' + str(oslot)
      output_dim = self._oslot_to_shape[oslot][-1] # TODO: This is only valid for 1D 
      hid_layer = fully_connected(x_in, num_nodes, activation_fn=activation)
      for _ in range(num_layers-1):
        hid_layer = fully_connected(hid_layer, int(num_nodes*net_grow_rate))
      output = fully_connected(hid_layer, output_dim)
      
      self._oslot_to_otensor[oslot] = tf.identity(output, output_name) 
      
    self._is_built = True
