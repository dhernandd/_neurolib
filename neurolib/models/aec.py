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
from neurolib.encoder.encoder import EncoderFactory
from neurolib.trainers.trainer import GDTrainer
from neurolib.utils.graphs import get_session
from neurolib.utils.utils import make_var_name



class BasicAE(Model):
  """
  """
  def __init__(self, ydim, xdim, specs):
    """
    PHILOSOPHY: Classes that inherit from Model are those that will be seen by
    the user. For the moment, I have decided that the tensorflow graph is
    created upon initialization. Hence, this classes should implement at least
    the following methods:
    
    train(...)
    sample(...)
    
    Moreover, as the project grows, other abstract subclasses will likely appear
    between the Model class and concrete classes such as this one that give rise
    to a more hierarchical structure of statistical models.
    
    
    TODO: Need to worry about dealing with separate models cocnurrently which
    requires handling multiple active graphs
    """
    # TODO: Decide how to pass these things
    
    # TODO: The idea here is that there are some hyperparameters that the user
    # MUST provide. Those should not be in the dictionary of specs, rather the
    # user should be forced to provide them on initialization.
    self.ydim = ydim
    self.xdim = xdim
    super(BasicAE, self).__init__()
    self.specs = specs
    
    # The main scope for this model. 
    
    # TODO: There is a design problem here if say, we want to work with 2
    # different models of this class. One possibility is to embed all this in a
    # Singleton class that handles things such as providing the right scope for
    # each model. This shouldn't be hard to implement.
    self.main_scope = 'BasicAE'
    
    # Build the model
    self._build()
    
  def _build(self):
    """
    TODO: Implement different options for Trainer
    """
    specs = self.specs
    ydim = self.ydim
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
    
    enc_fac = EncoderFactory()
    with tf.variable_scope(self.main_scope, reuse=tf.AUTO_REUSE):
      self.Y = tf.placeholder(dtype=tf.float32, shape=[b_size, ydim],
                              name='Input')
      self.enc_H1 = enc_fac(self.Y, enc1_specs, out_name='H1')
      self.enc_Yprime = enc_fac(self.enc_H1, enc2_specs, out_name='Yprime')
      
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
    """
    Y = self.Y
    Yprime = self.enc_Yprime.get_output()
    
    return tf.reduce_sum(Y*Yprime)

  def train(self, ytrain, num_epochs, yvalid=None):
    """
    Delegate training to the Trainer object
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
    name = make_var_name(self.main_scope, 'samp_'+'H1')
    
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
                          'activation' : tf.nn.relu, 'net_grow_rate' : 0.5,
                          'out_dim' : self.xdim, 'ipt_dim' : self.ydim}
    enc2_default_specs = {'num_layers' : 2, 'num_nodes' : 64,
                          'activation' : tf.nn.relu, 'net_grow_rate' : 2,
                          'out_dim' : self.ydim, 'ipt_dim' : self.xdim}
    
    if 'enc1_specs' not in specs: specs['enc1_specs'] = {}
    if 'enc2_specs' not in specs: specs['enc2_specs'] = {}
    enc1_default_specs.update(specs['enc1_specs'])
    enc2_default_specs.update(specs['enc2_specs'])

    return enc1_default_specs, enc2_default_specs
    