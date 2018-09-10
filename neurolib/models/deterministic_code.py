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
# import numpy as np
import tensorflow as tf

from neurolib.models.models import Model
# from neurolib.encoder.encoder import ( DeterministicEncoder, NormalEncoder, Encoder)
# from neurolib.encoder.encoder import EncoderNode
# from neurolib.encoder.deterministic import DeterministicEncoding
# from neurolib.encoder.normal import NormalEncoding

from neurolib.trainers.trainer import GDBender
from neurolib.utils.graphs import get_session
# from neurolib.utils.utils import make_var_name
from neurolib.builders.static_builder import StaticModelBuilder


class DeterministicCode(Model):
  """
  The DeterministicCode Model is the simplest possible model in the Encoder
  paradigm. It consists of a single encoder with a single input and output.
  
  in => E => out
  """
  def __init__(self, input_dim=None, output_dim=None, directives={}, builder=None):
    """
    """
    self.input_dim = input_dim
    self.output_dim = output_dim
    self._update_default_directives(directives)
    # The main scope for this model. 
    self.main_scope = 'DeterministicCode'

    super(DeterministicCode, self).__init__()
    self.builder = builder
    if builder is not None:
      self._check_build()
    else:
      if input_dim is None or output_dim is None:
        raise ValueError("Both the input dimension (in_dims) and the output dimension "
                         "(out_dims) are necessary in order to specify build the default "
                         "DeterministicCode.")
                    
  def _check_build(self):
    """
    A function to check that the client-provided builder corresponds indeed to a
    DeterministicCode. Not clear whether this is actually needed, keep for now.
    """
    pass
  
  def _update_default_directives(self, directives):
    """
    Updates the default specs with the ones provided by the user.
    """
    self.directives = {"nnodes_1stlayer" : 128,
                       "net_grow_rate" : 1.0,
                       "inner_activation_fn" : 'relu',
                       "output_activation_fn" : 'linear',
                       "class" : 'Deterministic'}
    self.directives.update(directives)
    
  def _get_directives(self):
    """
    Returns two directives to build the two encoders that make up the default
    Model.
    
    Directives:
      nnodes_1stlayer_encj : The number of nodes in the first hidden layer of
            the encoder j
      ntwrk_grow_rate_encj : The ratio between the number of nodes of subsequent
            layers
      TODO: ...
    """
    enc_directives = self.directives
    in_directives = {}
    out_directives = {}
    return enc_directives, in_directives, out_directives

  def build(self):
    """
    Builds the DeterministicCode.
    
    => E =>
    """
    builder = self.builder
    if builder is None:
      builder = StaticModelBuilder()
      
      enc_dirs, in_dirs, out_dirs = self._get_directives()

      in0 = builder.addInput(self.input_dim)
      in1 = builder.addInput(self.output_dim, directives=in_dirs)
      enc1 = builder.addInner(self.output_dim, directives=enc_dirs)
      out0 = builder.addOutput(directives=out_dirs)
      out1 = builder.addOutput(directives=out_dirs)
      
      builder.addDirectedLink(in0, enc1)
      builder.addDirectedLink(enc1, out0)
      builder.addDirectedLink(in1, out1)
    
      self._adj_list = builder.adj_list

    with tf.variable_scope(self.main_scope, reuse=tf.AUTO_REUSE):
      # Build the tensorflow graph
      builder.build()
      self.model_graph = builder.model_graph
      
      self.inputs.update(builder.input_nodes)
      self.outputs.update(builder.output_nodes)
      
      self.cost = self._define_cost()
      
    self._is_built = True
    self.bender = GDBender(self.cost)

  def _define_cost(self):
    """
    """
    outputs = list(self.outputs.values())
    O0 = outputs[0].get_inputs()[0]
    O1 = outputs[1].get_inputs()[0]
   
    return tf.reduce_sum((O0-O1)**2, name='mse_cost')
    
  def get_inputs(self):
    """
    """
    return self.inputs
  
  def get_outputs(self):
    """
    """
    return self.outputs
  
  def update(self, dataset):
    """
    """
    self.bender.update(dataset)
  
  def train(self, dataset, num_epochs=100):
    """
    Transforms a dataset provided by the user into a tensorflow feeddict. Then
    trains the model. 
    
    The dataset provided by the client should have keys
    
    train_#
    valid_#
    
    where # is the number of the corresponding Input node, see model graph.
    """
    self._check_dataset_correctness(dataset)

    def split_dataset(dataset):
      """
      """
      train_dataset = {}
      valid_dataset = {}
      for key in dataset:
        if key.startswith('train'):
          train_dataset[key] = dataset[key]
        elif key.startswith('train') == 'valid':
          valid_dataset[key] = dataset[key]
        else:
          raise KeyError("The dataset contains the key ", key, ". The prefixes of the "
                         "keys of the dataset must be either 'train' or 'valid'")
      return train_dataset, valid_dataset
    
    train_dataset, valid_dataset = split_dataset(dataset)

    self.bender.train(train_dataset, valid_dataset, scope=self.main_scope, num_epochs=num_epochs)
    
  def visualize_model_graph(self):
    """
    """
    self.model_graph.write_png("model_graph")
    
  def _check_dataset_correctness(self, dataset):
    """
    """
    pass
  
  
  
  
