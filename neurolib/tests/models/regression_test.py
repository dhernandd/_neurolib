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
import os
os.environ['PATH'] += ':/usr/local/bin'

import unittest

import numpy as np
import tensorflow as tf

from neurolib.models.regression import NeuralNetRegression
from neurolib.builders.static_builder import StaticModelBuilder

def make_data_iterator(data, batch_size=1, shuffle=True):
    """
    A simple data iterator
    """
    nsamps = len(data[0])
    l_inds = np.arange(nsamps)
    if shuffle: 
        np.random.shuffle(l_inds)
    
    for i in range(0, nsamps, batch_size):
        yield [ d[l_inds[i:i+batch_size]] for d in data ]
        
class NNRegressionFullTest(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()

  @unittest.skipIf(False, "Skipping")
  def test_init(self):
    """
    """
    print("\nTest 0: NNRegression initialization")
    NeuralNetRegression(input_dim=10, output_dim=1)
    
  @unittest.skipIf(False, "Skipping")
  def test_build(self):
    """
    """
    dc = NeuralNetRegression(input_dim=10, output_dim=1)
    dc.build()
#     dc.visualize_graph()
    
  @unittest.skipIf(True, "Skipping")
  def test_train(self):
    """
    """
    x = 10.0*np.random.randn(100, 2)
    y = x[:,0:1] + 1.5*x[:,1:]# + 3*x[:,1:]**2 + 0.5*np.random.randn(100,1)
    dataset = {'train_features' : x,
               'train_input_response' : y}
    
    dc = NeuralNetRegression(input_dim=2, output_dim=1)
    dc.build()
    dc.train(dataset, num_epochs=500)
  
  @unittest.skipIf(False, "Skipping")
  def test_train_custom_builder(self):
    """
    """
    x = 10.0*np.random.randn(100, 2)
    y = x[:,0:1] + 1.5*x[:,1:]# + 3*x[:,1:]**2 + 0.5*np.random.randn(100,1)
    dataset = {'train_features' : x,
               'train_input_response' : y}
    
    # DEFINE A BUILDER    
    builder = StaticModelBuilder()
    enc_dirs = {'num_layers_0' : 2,
                'num_nodes_0' : 128,
                'activation_0' : 'leaky_relu',
                'net_grow_rate_0' : 1.0 }
    in_dirs, out_dirs = {}, {}
    input_dim, output_dim = 2, 1
    in0 = builder.addInput(input_dim, name="features")
    enc1 = builder.addInner(10, directives=enc_dirs)
    enc2 = builder.addInner(output_dim, directives=enc_dirs)
    out0 = builder.addOutput(directives=out_dirs, name="prediction")

    in1 = builder.addInput(output_dim, directives=in_dirs, name="input_response")
    out1 = builder.addOutput(directives=out_dirs, name="response")
    
    builder.addDirectedLink(in0, enc1)
    builder.addDirectedLink(enc1, enc2)
    builder.addDirectedLink(enc2, out0)
    builder.addDirectedLink(in1, out1)
    
    # PASS IT TO THE MODEL INSTEAD OF THE DEFAULT
    tf.reset_default_graph()
    dc = NeuralNetRegression(input_dim=2, output_dim=1, builder=builder)
    dc.build()
    dc.visualize_model_graph("two_encoders.png")
    dc.train(dataset, num_epochs=10)
    

if __name__ == '__main__':
  tf.test.main()
  
