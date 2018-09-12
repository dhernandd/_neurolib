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

from neurolib.models.models import Model

from neurolib.trainers.trainer import GDBender
from neurolib.builders.static_builder import StaticModelBuilder


class NeuralNetRegression(Model):
  """
  The NeuralNetRegression Model is the simplest possible model in the Encoder
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
    self._main_scope = 'NeuralNetRegression'

    super(NeuralNetRegression, self).__init__()
    self.builder = builder
    if builder is not None:
      self._help_build()
    else:
      if input_dim is None or output_dim is None:
        raise ValueError("Both the input dimension (in_dims) and the output dimension "
                         "(out_dims) are necessary in order to specify build the default "
                         "NeuralNetRegression.")
                      
  def _help_build(self):
    """
    A function to check that the client-provided builder corresponds indeed to a
    NeuralNetRegression. Not clear whether this is actually needed, keep for now.
    """
    dirs = self.directives
    trainer = dirs['trainer']
    print("Hi! I see you are attempting to build a Regressor by yourself."
          "In order for your model to be consistent with the ", trainer,
          " Trainer, you must implement the following Output Nodes:")
    if trainer == 'gd-mse':
      print("OutputNode(input_dim={}, name='regressors')".format(self.output_dim))
      print("OutputNode(input_dim={}, name='response')".format(self.output_dim))
      print("\nThis is an absolute minimum requirement and NOT a guarantee that a custom "
            "model will be successfully trained (read the docs for more).")
      
  def _check_build(self):
    """
    """
    pass
    
  
  def _update_default_directives(self, directives):
    """
    Updates the default specs with the ones provided by the user.
    """
    self.directives = {'num_layers_0' : 2,
                       'num_nodes_0' : 128,
                       'activation_0' : 'leaky_relu',
                       'net_grow_rate_0' : 1.0,
                       'share_params' : False,
                       'trainer' : 'gd-mse',
                       'gd_optimizer' : 'adam'}
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
    Builds the NeuralNetRegression.
    
    => E =>
    """
    builder = self.builder
    if builder is None:
      builder = StaticModelBuilder()
      
      enc_dirs, in_dirs, out_dirs = self._get_directives()

      in0 = builder.addInput(self.input_dim, name="features")
      enc1 = builder.addInner(self.output_dim, directives=enc_dirs)
      out0 = builder.addOutput(directives=out_dirs, name="prediction")

      in1 = builder.addInput(self.output_dim, directives=in_dirs, name="response")
      out1 = builder.addOutput(directives=out_dirs, name="response")
      
      builder.addDirectedLink(in0, enc1)
      builder.addDirectedLink(enc1, out0)
      builder.addDirectedLink(in1, out1)
    
      self._adj_list = builder.adj_list
    else:
      self._check_build()

    with tf.variable_scope(self.main_scope, reuse=tf.AUTO_REUSE):
      # Build the tensorflow graph
      builder.build()
      self.model_graph = builder.model_graph
      
      for node in builder.input_nodes.values():
        self.inputs[node.name] = node.get_outputs()[0]
#         self.inputs.update(builder.input_nodes)
      for node in builder.output_nodes.values():
        self.outputs[node.name] = node.get_inputs()[0]
      
      self.cost = self._define_cost()
      
    self._is_built = True
    self.bender = GDBender(self.cost)

  def _define_cost(self):
    """
    """
    O0 = self.outputs["prediction"]
    O1 = self.outputs["response"]
   
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
    Trains the model. 
    
    The dataset provided by the client should have keys
    
    train_features, train_response
    valid_features, valid_response
    test_features, test_response
    
    where # is the number of the corresponding Input node, see model graph.
    """
    self._check_dataset_correctness(dataset)

    train_dataset, valid_dataset, _ = self.make_datasets(dataset)
    self.bender.train(train_dataset, valid_dataset, scope=self.main_scope, num_epochs=num_epochs)
    
  def visualize_model_graph(self, filename="model_graph"):
    """
    Generates a representation of the computational graph
    """
    self.model_graph.write_png(filename)
    
  def _check_dataset_correctness(self, dataset):
    """
    """
    pass
  
  def get_encoders(self):
    """
    """
    pass

class BayesianRegression(Model):
  """
  """
  def __init__(self):
    """
    """
    pass
  
  
  
  
  
  
