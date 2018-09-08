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

from neurolib.models.models import Model
# from neurolib.builders.aec import AECBuilder
from neurolib.encoder.encoder import ( DeterministicEncoder, NormalEncoder, Encoder)
from neurolib.encoder.encoder_factory import EncoderFactory
from neurolib.trainers.trainer import GDTrainer
from neurolib.utils.graphs import get_session
from neurolib.utils.utils import make_var_name
from neurolib.builders.builder import StaticModelBuilder


class DeterministicCode(Model, Encoder, StaticModelBuilder):
  """
  The DeterministicCode Model is the simplest possible model in the Encoder
  paradigm. It consists of a single encoder with a single input and output.
  
  in => E => out
  """
  def __init__(self, in_dims=None, out_dims=None, specs={}, builder=None):
    """
    """
    # The main scope for this model. 
    self.main_scope = 'DeterministicCode'

    if builder is not None:
      super(DeterministicCode, self).__init__(builder=builder)
      self._check_build()
      self._build_from_builder()
    else:
      if in_dims is None or out_dims is None:
        raise ValueError("Both the input dimension (in_dims) and the output dimension "
                         "(out_dims) are necessary in order to build the "
                         "DeterministicCode")
      else:
        super(DeterministicCode, self).__init__(in_dims=in_dims, out_dims=out_dims,
                                                    specs=specs)
        self.builder = StaticModelBuilder()

        self._build_default() 
  
  def _check_build(self):
    """
    A function to check that the user provided builder corresponds indeed to a
    DeterministicCode. Not clear whether this is actually needed, keep for now.
    """
    pass
  
  def _get_default_directives(self):
    """
    Updates the default specs with the ones provided by the user. Returns two
    directives to build the two encoders that make up the default Model.
    
    Directives:
      nnodes_1stlayer_encj : The number of nodes in the first hidden layer of
            the encoder j
      ntwrk_grow_rate_encj : The ratio between the number of nodes of subsequent
            layers
      TODO: ...
    """
    default_specs = {"nnodes_1stlayer" : 128,
                     "ntwrk_grow_rate" : 0.5,
                     "inner_activation_fn" : 'relu',
                     "output_activation_fn" : 'linear',
                     "class" : 'Deterministic',
                     "in_dims" : self.in_dims,
                     "out_dims" : self.out_dims}
    default_specs.update(self.specs)
    
    enc_directives = default_specs
    in_directives = {"in_dims" : self.in_dims}
    out_directives = {"out_dims" : self.out_dims}
    return enc_directives, in_directives, out_directives

  def _build_default(self):
    """
    Builds the default DeterministicCode
    
    => E =>
    """
    builder = self.builder
    enc1_dirs, in_dirs, out_dirs = self._get_default_directives()
    with tf.variable_scope(self.main_scope, reuse=tf.AUTO_REUSE):
      in1 = builder.addInput(in_dirs)
      enc1 = builder.addEncoder(enc1_dirs)
      out1 = builder.addOutput(out_dirs)
      builder.addDirectedLink(in1, enc1)
      builder.addDirectedLink(enc1, out1)
      
      # This line build the tensorflow graph
      self.model = builder.build()
  
class LinearDeterministicEncoding(Model):
  """
  
  in => E1 => E2 => ... => En => out
  
  """
  def __init__(self, ydim=None, xdim=None, specs={}, builder=None):
    """
    Any Model, and in particular the BasicEncoder, can be initialized in (at the
    moment) 2 different ways: the default way and through a previously defined
    builder object. The default initialization provides a quick broad way of
    specifying a basic Model and getting results. For a lot more flexibility,
    initialization through the Builder is appropriate.
    
    TODO: Need to worry about dealing with separate models concurrently which
    requires handling multiple active graphs
    """
    # The main scope for this model. 
    self.main_scope = 'LinDetEnc'
    
    # THINK: There is a light design issue here. Say, we want to work with 2
    # different models of this class. One possibility is to embed all this in a
    # Singleton class that handles things such as providing the right scope for
    # each model. This shouldn't be too hard to implement. 
    
    # ON THE OTHER HAND, it may be unwise to allow the user to define a disjoint
    # tensorflow Graph, so I will ignore this for now.
    if builder is not None:
      super(LinearDeterministicEncoding, self).__init__(builder=builder)
      self._build_from_builder()
    else:
      if ydim is None or xdim is None:
        raise ValueError("Both the input dimension (ydim) and the output dimension "
                         "(xdim) are necessary in order to build the "
                         "default LinearDeterministicEncoding")
      else:
        super(LinearDeterministicEncoding, self).__init__(in_dims=ydim, out_dims=xdim,
                                                          specs=specs)
        self._build_default()      
              
  def _get_default_directives(self):
    """
    Updates the default specs with the ones provided by the user. Returns two
    directives to build the two encoders that make up the default Model.
    
    Directives:
      nnodes_1stlayer_encj : The number of nodes in the first hidden layer of
            the encoder j
      ntwrk_grow_rate_encj : The ratio between the number of nodes of subsequent
            layers
      TODO: ...
    """
    default_specs = {"nnodes_1stlayer_1" : 128,
                     "nnodes_1stlayer_2" : 128, 
                     "ntwrk_grow_rate_1" : 0.5,
                     "ntwrk_grow_rate_2" : 2.0,
                     "inner_activation_fn_1" : 'relu',
                     "inner_activation_fn_2" : 'relu',
                     "output_activation_fn_1" : 'linear',
                     "output_activation_fn_2" : 'linear',
                     "class_1" : 'Deterministic',
                     "class_2" : 'Deterministic'}
    default_specs.update(self.specs)
    
    # Removes the last two characters from the keys of the Model specs
    # dictionary to form the encoder directives.
    enc1_directives = { key[:-2] : value for key, value in default_specs.items() if
                        key[-1] == '1'}
    enc2_directives = { key[:-2] : value for key, value in default_specs.items() if
                        key[-1] == '2'}
    return enc1_directives, enc2_directives
    
  def _build_default(self):
    """
    Builds the default model
    
    => E1 => E2 =>
    """
    enc1_directives, enc2_directives = self._get_default_directives()
    with tf.variable_scope(self.main_scope, reuse=tf.AUTO_REUSE):
      builder = StaticModelBuilder()
      enc1 = builder.addEncoder(enc1_directives)
      enc2 = builder.addEncoder(enc2_directives)
      builder.addDirectedLink('in', enc1)
      builder.addDirectedLink(enc1, enc2)
      builder.addDirectedLink(enc1, 'out')
      
      self.model = builder.build()
    
  def _build_from_builder(self):
    """
    """
    self.model = self.builder.build()
    
  def _build(self):
    """
    TODO: Implement different options for Trainer
    """
    specs = self.specs
    b_size = specs['b_size'] if 'b_size' in self.specs else 1
        
    # These line builds the specs that will be used to construct the Encoders.
    # Each Encoder object is an abstraction of the sequence:
    #
    # Code => Mapping => Code
    enc1_specs, enc2_specs =  self._build_encoder_specs()
    
    # TODO: Then there are the Trainer objects. This takes into account the
    # possibility that we may want to train models in different ways. In the
    # case of this simple AutoEncoder, the training procedure is just gradient
    # descent on a cost function. This training strategy is implemented in
    # GDTrainer which takes a cost and some specs.
    train_specs = {'lr' : 1e-6, 'optimizer' : 'adam'}
    
    # TODO: Encapsulate from here in a function _stack_encoders() that puts the
    # bricks toegether
    enc_fac = EncoderFactory()
    with tf.variable_scope(self.main_scope, reuse=tf.AUTO_REUSE):
      self.Y = tf.placeholder(dtype=tf.float32, shape=[b_size, self.ydim],
                              name='Fac'+self.linearized_factor_ids.pop(0))
      self.enc_H1 = enc_fac(self.Y, DeterministicEncoder, enc1_specs,
                            output_name='Fac'+self.linearized_factor_ids.pop(0))
      self.enc_Yprime = enc_fac(self.enc_H1, DeterministicEncoder, enc2_specs,
                            output_name='Fac'+self.linearized_factor_ids.pop(0))
      
      # TODO: Note the use of node_names below. Nodes correspond to the codes of
      # the different Encoder objects and I am foreseeing to implement an
      # automatic assignment of integers to the nodes as their names. The node
      # names are used by the tensorflow session that is run during training to
      # feed the proper tensors. Any idea to make this more elegant is
      # appreciated.
      self.cost = self._build_cost()
      self.trainer = GDTrainer(self.cost, train_specs,
                               node_names={0 : self.Y.name})
#       self.output = AECBuilder.build(self.specs) 
    
  def _build_cost(self):
    """
    TODO: Normalize the cost per sample, etc.
    """
    Y = self.Y
    Yprime = self.enc_Yprime.get_output()
    
    return tf.reduce_sum(Y*Yprime)

  def train(self, ytrain, num_epochs, yvalid=None):
    """
    Delegate training to the Trainer object.
    """
    self.trainer.train(ytrain, num_epochs=num_epochs, yvalid=yvalid)
    
  def sample(self, num_samples=100, input_nodes=None, output_nodes=None):
    """
    TODO: This is VERY preliminary and inflexible!
    
    TODO: Deal with lists of input and output nodes.
    
    TODO: Close and open sessions using the context manager? Again, need to
    think about the most elegant way to deal with this
    
    PHILOSOPHY: There should be a simple sample function, the most common one
    that we know people are going to use. More complicated sampling should be
    handled elsewhere.
    """
    sess = get_session()
    xsamps = np.random.randn(num_samples, self.xdim)
    name = make_var_name(self.main_scope, 'samp_'+'Fac1')
    
    sess.run(tf.global_variables_initializer())
    ysamps = sess.run(self.enc_Yprime.sample_out, feed_dict={name : xsamps})
    sess.close()
    
    return ysamps
    
  def _build_encoder_specs(self):
    """
    Implemented specs:
      activation
      net_grow_rate
      num_layers
      num_nodes
      out_dim
    
    """
    specs = self.specs
    enc1_default_specs = {'num_layers' : 2, 'num_nodes' : 128,
                          'activation' : tf.nn.leaky_relu, 'net_grow_rate' : 0.5,
                          'output_dim' : self.xdim, 'ipt_dim' : self.ydim}
    enc2_default_specs = {'num_layers' : 2, 'num_nodes' : 64,
                          'activation' : tf.nn.leaky_relu, 'net_grow_rate' : 2,
                          'output_dim' : self.ydim, 'ipt_dim' : self.xdim}
    
    if 'enc1_specs' not in specs: specs['enc1_specs'] = {}
    if 'enc2_specs' not in specs: specs['enc2_specs'] = {}
    enc1_default_specs.update(specs['enc1_specs'])
    enc2_default_specs.update(specs['enc2_specs'])

    return enc1_default_specs, enc2_default_specs
    
    
class BayesianAE(Model):
  """
  For lack of a better name...
  """
  def __init__(self, ydim, xdim, specs):
    """
    Read the comments for the BasicEncoder class 
    """
    self._init_specs(specs)
    super(BayesianAE, self).__init__(specs)

    self.ydim = ydim
    self.xdim = xdim
    self.main_scope = 'BayesianAE'
    
    # Build the model
    self._build()

  def _build(self):
    """
    """
    specs = self.specs
    b_size = specs['b_size'] if 'b_size' in self.specs else 1
    
    # These line builds the specs that will be used to construct the Encoders.
    # Each Encoder object is an abstraction of the sequence:
    #
    # Code => Mapping => Code
    enc1_specs, enc2_specs =  self._build_encoder_specs()
    
    # TODO: Then there are the Trainer objects. This takes into account the
    # possibility that we may want to train models in different ways. In the
    # case of this simple AutoEncoder, the training procedure is just gradient
    # descent on a cost function. This training strategy is implemented in
    # GDTrainer which takes a cost and some specs.
    train_specs = {'lr' : 1e-9, 'optimizer' : 'adam'}
    
    enc_fac = EncoderFactory()
    with tf.variable_scope(self.main_scope, reuse=tf.AUTO_REUSE):
      self.Y = tf.placeholder(dtype=tf.float32, shape=[b_size, self.ydim],
                              name='Fac'+self.linearized_factor_ids.pop(0))
      self.enc_H1 = enc_fac(self.Y, NormalEncoder, enc1_specs,
                            output_name='Fac'+self.linearized_factor_ids.pop(0))
      self.enc_Yprime = enc_fac(self.enc_H1, NormalEncoder, enc2_specs,
                            output_name='Fac'+self.linearized_factor_ids.pop(0))
      
      # TODO: Note the use of node_names below. Nodes correspond to the codes of
      # the different Encoder objects and I am foreseeing to implement an
      # automatic assignment of integers to the nodes as their names. The node
      # names are used by the tensorflow session that is run during training to
      # feed the proper tensors. Any idea to make this more elegant is
      # appreciated.
      self.cost = self._build_cost()
      self.trainer = GDTrainer(self.cost, train_specs,
                               node_names={0 : self.Y.name})
#       self.output = AECBuilder.build(self.specs) 

  def _build_cost(self):
    """
    TODO: Normalize the cost per sample, etc.
    """
    return -self.enc_Yprime.dist.log_prob(self.Y)

  def _init_specs(self, specs):
    """
    """
    specs['num_factors'] = 3

  def train(self, ytrain, num_epochs, yvalid=None):
    """
    """
    self.trainer.train(ytrain, num_epochs=num_epochs, yvalid=yvalid)

  def sample(self, num_samples=100, input_nodes=None, output_nodes=None):
    """
    TODO: This is VERY preliminary and inflexible!
    
    TODO: Deal with lists of input and output nodes.
    
    TODO: Close and open sessions using the context manager? Again, need to
    think about the most elegant way to deal with this
    
    PHILOSOPHY: There should be a simple sample function, the most common one
    that we know people are going to use. More complicated sampling should be
    handled elsewhere.
    """
    sess = get_session()
    xsamps = np.random.randn(num_samples, self.xdim)
    name = make_var_name(self.main_scope, 'samp_'+'Fac1')
    
    sess.run(tf.global_variables_initializer())
    ysamps = sess.run(self.enc_Yprime.sample_out, feed_dict={name : xsamps})
    sess.close()
    
    return ysamps
  
  def _build_encoder_specs(self):
    """
    Implemented specs:
      activation
      net_grow_rate
      num_layers
      num_nodes
      out_dim
    """
    specs = self.specs
    enc1_default_specs = {'num_layers' : 2, 'num_nodes' : 128,
                          'activation' : tf.nn.leaky_relu, 'net_grow_rate' : 0.5,
                          'output_dim' : self.xdim, 'ipt_dim' : self.ydim}
    enc2_default_specs = {'num_layers' : 2, 'num_nodes' : 64,
                          'activation' : tf.nn.leaky_relu, 'net_grow_rate' : 2,
                          'output_dim' : self.ydim, 'ipt_dim' : self.xdim}
    
    if 'enc1_specs' not in specs: specs['enc1_specs'] = {}
    if 'enc2_specs' not in specs: specs['enc2_specs'] = {}
    enc1_default_specs.update(specs['enc1_specs'])
    enc2_default_specs.update(specs['enc2_specs'])

    return enc1_default_specs, enc2_default_specs
    
    