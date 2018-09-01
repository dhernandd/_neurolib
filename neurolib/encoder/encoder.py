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

# TODO: Import all these!
from tensorflow.contrib.layers.python.layers import fully_connected

class EncoderFactory():
  """
  """
#   def __init__(self, scope):
#     """
#     """
      
  def __call__(self, ipt, specs, out_name):
    """
    """
    # Allow for both a tf.Tensor or an Encoder be provided as ipt.
    if isinstance(ipt, Encoder):
      ipt = ipt.get_output()
      
    if isinstance(specs, dict):
      encoder = Encoder(ipt, specs, out_name)
    else:
      # TODO: Add other possible ways of passing specs!
      pass
    return encoder
    

class Encoder():
  """
  """
  def __init__(self, ipt, specs, out_name):
    """
    TODO: The user should be able to pass a tensorflow graph. In that case,
    Encoder should act as a simple wrapper that returns the input and the output.
    """
    self.ipt = ipt
    ipt_name = ipt.name.split('/')[-1][:-2] 
    self.out_name = out_name
    self.specs = specs
    
    self.out = self.build_encoder_graph()
    
    # TODO: This way of creating samples is a quick fix. Make it flexible
    # (arbitrary rank tensors, etc)
    sample_in = tf.placeholder(tf.float32, shape=[None, specs['ipt_dim']],
                                 name='samp_'+ipt_name)
    self.sample_out = self.build_encoder_graph(sample_in,
                                               out_name=self.out_name + '_samp')
          
    
  def get_input(self):
    """
    """
    return self.ipt
    
  def get_output(self):
    """
    """
    return self.out
    
  def build_encoder_graph(self, ipt=None, out_name=None):
    """
    Builds the graph corresponding to a single encoder.
    
    TODO: Expand this a lot, many more specs necessary.
    """
    if ipt is None: ipt = self.ipt
    if out_name is None: out_name = self.out_name
    directives = self.specs
    
    num_layers = directives['num_layers']
    num_nodes = directives['num_nodes']
    activation = directives['activation']
    net_grow_rate = directives['net_grow_rate']
    out_dim = directives['out_dim']
    
    hid_layer = fully_connected(ipt, num_nodes, activation_fn=activation)
    for _ in range(num_layers-1):
      hid_layer = fully_connected(hid_layer, int(num_nodes*net_grow_rate))
    output = fully_connected(hid_layer, out_dim)
    
    return tf.identity(output, self.out_name) 

