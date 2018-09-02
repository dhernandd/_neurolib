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
import abc
from collections import namedtuple
DistPars = namedtuple("DistPars", ("distribution", "get_out_dims"))

import tensorflow as tf

# TODO: Import all these!
from tensorflow.contrib.layers.python.layers import fully_connected

# TODO: import all distributions, provide support for all of them

class EncoderFactory():
  """
  """
#   def __init__(self, scope):
#     """
#     """
      
  def __call__(self, ipt, EncoderClass, specs, output_name):
    """
    """
    # Allow for both a tf.Tensor or an Encoder be provided as input.
    if isinstance(ipt, Encoder):
      ipt = ipt.get_output()
      
    if isinstance(specs, dict):
      encoder = EncoderClass(ipt, specs, output_name)
    else:
      # TODO: Add other possible ways of passing specs!
      pass
    return encoder
    

class Encoder(abc.ABC):
  """
  """
  def __init__(self, ipt, specs, output_name):
    """
    TODO: The user should be able to pass a tensorflow graph directly. In that
    case, Encoder should act as a simple wrapper that returns the input and the
    output.
    """
    self.input = ipt 
    self.output_name = output_name
    self.specs = specs

  @abc.abstractmethod
  def _build_encoder_graph(self,  ipt=None, output_name=None):
    """
    This method must define all relevant variables for this encoder and return
    the encoder output
    """
    raise NotImplementedError("Please implement me.")
  
  def get_input(self):
    """
    """
    return self.input
    
  def get_output(self):
    """
    """
    return self.output


class DeterministicEncoder(Encoder):
  """
  """
  def __init__(self, ipt, specs, output_name):
    """
    TODO: The user should be able to pass a tensorflow graph directly. In that
    case, Encoder should act as a simple wrapper that returns the input and the
    output.
    """
    super(DeterministicEncoder, self).__init__(ipt, specs, output_name)
    
    self.output = self._build_encoder_graph()
    
    # TODO: This way of creating samples is a quick fix. Make it flexible
    # (arbitrary rank tensors, etc)
    input_name = ipt.name.split('/')[-1][:-2]
    sample_in = tf.placeholder(tf.float32, shape=[None, specs['ipt_dim']],
                                name='samp_'+input_name)
    self.sample_out = self._build_encoder_graph(sample_in,
                                output_name=self.output_name + '_samp')
    
  def _build_encoder_graph(self, ipt=None, output_name=None):
    """
    Builds the graph corresponding to a single encoder.
    
    TODO: Expand this a lot, many more specs necessary.
    
    TODO: Think of the right way to organize this, how much can be done in the
    base class?
    """
    if ipt is None: ipt = self.input
    if output_name is None: output_name = self.output_name
    specs = self.specs

    num_layers = specs['num_layers']
    num_nodes = specs['num_nodes']
    activation = specs['activation']
    net_grow_rate = specs['net_grow_rate']
    out_dim = specs['out_dim']
    
    # The composition of layers that defines the encoder map 
    
    # TODO: Add capabilities for different types of layers, convolutional, etc.
    hid_layer = fully_connected(ipt, num_nodes, activation_fn=activation)
    for _ in range(num_layers-1):
      hid_layer = fully_connected(hid_layer, int(num_nodes*net_grow_rate))
    output = fully_connected(hid_layer, out_dim)
    
    return tf.identity(output, self.output_name) 


class BayesianEncoder():
  """
  """
  dist_dict = {'Normal' : DistPars(tf.distributions.Normal,
                                   [lambda x : x, lambda x : x**2])}
  
  def __init__(self, ipt, specs, out_name):
    """
    TODO: The user should be able to pass a tensorflow graph directly. In that
    case, Encoder should act as a simple wrapper that returns the input and the
    output.
    """
    self.input = ipt
    ipt_name = input.name.split('/')[-1][:-2] 
    self.output_name = out_name
    self.specs = specs
    
    self.output = self._build_encoder_graph()
    
    # TODO: This way of creating samples is a quick fix. Make it flexible
    # (arbitrary rank tensors, etc)
    sample_in = tf.placeholder(tf.float32, shape=[None, specs['ipt_dim']],
                                 name='samp_'+ipt_name)
    self.sample_out = self._build_encoder_graph(sample_in,
                                               output_name=self.output_name + '_samp')

  def get_input(self):
    """
    """
    return self.input
    
  def get_outputs(self):
    """
    """
    return [self.output]
    
  def _build_encoder_graph(self, ipt=None, out_names=None, distribution='Normal'):
    """
    Builds the graph corresponding to a single encoder.
    
    TODO: Expand this a lot, many more specs necessary.
    """
    if ipt is None: ipt = self.input
    if out_names is None: out_names = self.out_names
    directives = self.specs
    
    num_layers = directives['num_layers']
    num_nodes = directives['num_nodes']
    activation = directives['activation']
    net_grow_rate = directives['net_grow_rate']
    out_dim = directives['out_dim']

    Distribution = self.dist_dict[distribution].distribution
    get_out_dims = self.dist_dict[distribution].get_out_dims
    
    for func, out_name in zip(get_out_dims, out_names):
      hid_layer = fully_connected(ipt, num_nodes, activation_fn=activation)
      for _ in range(num_layers-1):
        hid_layer = fully_connected(hid_layer, int(num_nodes*net_grow_rate))
      output = fully_connected(hid_layer, func(out_dim))
      tf.identity(output, out_name)
    
    return 
