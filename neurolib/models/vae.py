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
from neurolib.models.models import Model

from neurolib.trainers.trainer import GDBender
from neurolib.builders.static_builder import StaticModelBuilder
from neurolib import cost_dict
from neurolib.encoder.normal import NormalTriLNode


class VariationalAutoEncoder(Model):
  """
  The Static Variational Autoencoder.   
  """
  def __init__(self, input_dim=None, output_dim=None, directives={}, builder=None):
    """
    """
    self.input_dim = input_dim
    self.output_dim = output_dim
    self._update_default_directives(directives)
    
    # The main scope for this model. 
    self._main_scope = 'VariationalAutoEncoder'

    super(VariationalAutoEncoder, self).__init__()
    self.builder = builder
    if builder is not None:
      self._help_build()
    else:
      if input_dim is None or output_dim is None:
        raise ValueError("Both the input dimension (in_dims) and the output dimension "
                         "(out_dims) are necessary in order to specify build the default "
                         "VariationalAutoEncoder.")
                      
  def _help_build(self):
    """
    A function to check that the client-provided builder corresponds indeed to a
    VariationalAutoEncoder. Not clear whether this is actually needed, keep for now.
    """
    dirs = self.directives
    trainer = dirs['trainer']
#     print("Hi! I see you are attempting to build a Regressor by yourself."
#           "In order for your model to be consistent with the ", trainer,
#           " Trainer, you must implement the following Output Nodes:")
#     if trainer == 'gd-mse':
#       print("OutputNode(input_dim={}, name='regressors')".format(self.output_dim))
#       print("OutputNode(input_dim={}, name='response')".format(self.output_dim))
#       print("\nThis is an absolute minimum requirement and NOT a guarantee that a custom "
#             "model will be successfully trained (read the docs for more).")
      
  def _update_default_directives(self, directives):
    """
    Updates the default specs with the ones provided by the user.
    """
    self.directives = {'num_layers_0' : 2,
                       'num_nodes_0' : 128,
                       'activation_0' : 'leaky_relu',
                       'net_grow_rate_0' : 1.0,
                       'share_params' : False,
                       'trainer' : 'gd',
                       'cost' : 'elbo',
                       'gd_optimizer' : 'adam',
                       'node_class' : NormalTriLNode}
    self.directives['cost'] = cost_dict[self.directives['cost']]  # @UndefinedVariable
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
    Builds the VariationalAutoEncoder.
    
    => E =>
    """
    dirs = self.directives
    builder = self.builder
    if builder is None:
      self.builder = builder = StaticModelBuilder(scope=self.main_scope)
      
#       i0 = builder.addInput(self.input_dim, name='gen_features', directives={})
      enc0 = builder.addInner(self.output_dim, name='Generative',
                              node_class=dirs['node_class'],
                              directives=dirs)
#       o0 = builder.addOutput(name='gen_response', directives={})
# 
      i1 = builder.addInput(self.output_dim, name='response', directives={})
      enc1 = builder.addInner(self.input_dim, name='Recognition',
                              node_class=dirs['node_class'],
                              directives=dirs)
      o1 = builder.addOutput(name='copy', directives={})

      builder.addDirectedLink(i1, enc1)
      builder.addDirectedLink(enc1, enc0, oslot=0)
#       builder.addDirectedLink(i1, enc1)
      builder.addDirectedLink(enc0, o1, oslot=0)      
    
      self._adj_list = builder.adj_list
    else:
      self._check_build()
      builder.scope = self.main_scope

    # Build the tensorflow graph
    self.nodes = self.builder.nodes
    builder._build()
    self.model_graph = builder.model_graph
    
    self.cost = self._define_cost()
      
    self.bender = GDBender(self.cost)
    
    self._is_built = True
    
  def _check_build(self):
    """
    """
    pass

  def _define_cost(self):
    """
    If I pass a distribution here, I can simply call inside of self.cost
    dist.entropy. + logp(dist.data, dist.sample). This raises a number of
    questions however
    
    1.- I had planned to make the cost a function of the outputs, not of the
    distributions
    
    2.- If I make it a function of the distributions, how is it extensible? That
    is, I am going to be able to compute loglikelihoods and entropies because
    the distributions I am dealing with are Gaussian, but as soon as they
    aren't, the cost in terms of them is not going to work anymore.
    
    3.- So, my take on this, I have to provide both 
    """
    cost = self.directives['cost']
    
    return cost(self.nodes)

  
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

    self.bender.train(train_dataset, valid_dataset, scope=self.main_scope,
                      num_epochs=num_epochs)
    
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


  
  
  
  
  
  
