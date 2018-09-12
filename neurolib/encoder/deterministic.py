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

from neurolib.encoder.encoder import InnerNode

act_fn_dict = {'relu' : tf.nn.relu,
               'leaky_relu' : tf.nn.leaky_relu}

class DeterministicEncoding(InnerNode):
  """
  """
  def __init__(self, label, output_shapes, name=None, directives={}):
    """
    TODO: The user should be able to pass a tensorflow graph directly. In that
    case, EncoderNode should act as a simple wrapper that returns the input and the
    output.
    """
    super(DeterministicEncoding, self).__init__(label, output_shapes, name=name)
    self._update_directives(directives)
        
  def _update_directives(self, directives):
    """
    """
    self.directives = {'num_layers_0' : 2,
                       'num_nodes_0' : 128,
                       'activation_0' : 'relu',
                       'net_grow_rate_0' : 1.0}
    self.directives.update(directives)
    
    # Deal with directives that should map to tensorflow objects hidden from the client
    self.directives['activation_0'] = act_fn_dict[self.directives['activation_0']]
    
  def _build(self, inputs=None):
    """
    Builds the graph corresponding to a single encoder.
    
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

    num_layers = dirs['num_layers_0']
    num_nodes = dirs['num_nodes_0']
    activation = dirs['activation_0']
    net_grow_rate = dirs['net_grow_rate_0']
  
    # The composition of layers that defines the encoder map. TODO: Add
    # capabilities for different types of layers, convolutional, etc. The main
    # issue is dealing with shape
    for oslot in range(self.num_outputs):
      output_dims = self.oslot_to_shape[oslot][0] # TODO: This is only valid for 1D 
      output_name = "Out_" + str(self.label) + '_' + str(oslot)
      hid_layer = fully_connected(x_in, num_nodes, activation_fn=activation)
      for _ in range(num_layers-1):
        hid_layer = fully_connected(hid_layer, int(num_nodes*net_grow_rate))
      output = fully_connected(hid_layer, output_dims)
      self.outputs[oslot] = tf.identity(output, output_name) 
      
    self._is_built = True
