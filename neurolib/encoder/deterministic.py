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

act_fn_dict = {'relu' : tf.nn.relu}

class DeterministicEncoding(InnerNode):
  """
  """
  def __init__(self, label, output_shapes, directives={}):
    """
    TODO: The user should be able to pass a tensorflow graph directly. In that
    case, EncoderNode should act as a simple wrapper that returns the input and the
    output.
    """
    super(DeterministicEncoding, self).__init__(label, output_shapes, directives)
    self._update_directives()
        
  def _update_directives(self):
    """
    """
    default_dirs = {'num_layers' : 2,
                    'num_nodes' : 128,
                    'activation' : 'relu',
                    'net_grow_rate' : 1.0}
    default_dirs.update(self.directives)
    self.directives = default_dirs
    
    self.directives['activation'] = act_fn_dict[self.directives['activation']]
    
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

    num_layers = dirs['num_layers']
    num_nodes = dirs['num_nodes']
    activation = dirs['activation']
    net_grow_rate = dirs['net_grow_rate']
  
    # The composition of layers that defines the encoder map. TODO: Add
    # capabilities for different types of layers, convolutional, etc. The main
    # issue is dealing with shape
    for j in range(self.num_outputs):
      output_dims = self.oslot_to_shape[j][0] # TODO: This is only valid for 1D 
      output_name = "DetEnc_" + str(self.label) + '_' + str(j)
      hid_layer = fully_connected(x_in, num_nodes, activation_fn=activation)
      for _ in range(num_layers-1):
        hid_layer = fully_connected(hid_layer, int(num_nodes*net_grow_rate))
      output = fully_connected(hid_layer, output_dims)
      self.outputs[j] = tf.identity(output, output_name) 
      
    self._is_built = True
