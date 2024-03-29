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

from neurolib.models.regression import Regression
from neurolib.builders.static_builder import StaticBuilder

# pylint: disable=bad-indentation, no-member, protected-access

NUM_TESTS = 3
test_to_run = 3 # np.random.choice(3)

class RegressionTestTrainCust(tf.test.TestCase):
  """
  """
  @unittest.skipIf(test_to_run != 1, "Skipping Test 0")
  def test_train_custom_builder(self):
    """
    """
    print("Test 0: Chain of Encoders\n")
    x = 10.0*np.random.randn(100, 2)
    y = x[:,0:1] + 1.5*x[:,1:] # + 3*x[:,1:]**2 + 0.5*np.random.randn(100,1)
    dataset = {'train_features' : x,
               'train_input_response' : y}
    
    # Define a builder.
    # More control on the directives of each node
    builder = StaticBuilder('CustBuild')
    enc_dirs = {'num_layers' : 2,
                'num_nodes' : 128,
                'activation' : 'leaky_relu',
                'net_grow_rate' : 1.0 }
    input_dim, output_dim = 2, 1
    in0 = builder.addInput(input_dim, name="features")
    enc1 = builder.addInner(1, 10, directives=enc_dirs)
    enc2 = builder.addInner(1, output_dim, directives=enc_dirs)
    out0 = builder.addOutput(name="prediction")

    builder.addDirectedLink(in0, enc1)
    builder.addDirectedLink(enc1, enc2)
    builder.addDirectedLink(enc2, out0)
    
    # Pass the builder object to the Regression Model
    reg = Regression(builder=builder)
    reg.build()
#     reg.visualize_model_graph("two_encoders.png")
    reg.train(dataset, num_epochs=50)

  @unittest.skipIf(test_to_run != 2, "Skipping Test 1")
  def test_train_custom_builder2(self):
    """
    Build a custom Regression model whose Model graph has the rhombic design
    """
    print("Test 1: Rhombic Design\n")
    x = 10.0*np.random.randn(100, 2)
    y = x[:,0:1] + 1.5*x[:,1:] # + 3*x[:,1:]**2 + 0.5*np.random.randn(100,1)
    dataset = {'train_features' : x,
               'train_input_response' : y}
    
    # Define a builder, control the graph
    # Like a lot more control...
    builder = StaticBuilder('CustBuild')
    enc_dirs = {'num_layers' : 2,
                'num_nodes' : 128,
                'activation' : 'leaky_relu',
                'net_grow_rate' : 1.0 }
    input_dim = 2
    in0 = builder.addInput(input_dim, name="features")
    enc1 = builder.addInner(10, num_inputs=1, directives=enc_dirs)
    enc2 = builder.addInner(10, num_inputs=1, directives=enc_dirs)
    enc3 = builder.addInner(1, num_inputs=2, num_layers=1, activation='linear')
    out0 = builder.addOutput(name="prediction")

    builder.addDirectedLink(in0, enc1)
    builder.addDirectedLink(in0, enc2)
    builder.addDirectedLink(enc1, enc3, islot=0)
    builder.addDirectedLink(enc2, enc3, islot=1)
    builder.addDirectedLink(enc3, out0)
    
    # Pass the builder object to the Regression Model
    reg = Regression(input_dim=2, output_dim=1, builder=builder)
    reg.build()
#     reg.visualize_model_graph("two_encoders.png")
    reg.train(dataset, num_epochs=50)
    
  @unittest.skipIf(test_to_run != 3, "Skipping")
  def test_train_custom_node2(self):
    """
    Test commit
    """
    print("Test 2: with CustomNode\n")
    input_dim = 2
    x = 10.0*np.random.randn(100, input_dim)
    y = x[:,0:1] + 1.5*x[:,1:] # + 3*x[:,1:]**2 + 0.5*np.random.randn(100,1)
    dataset = {'train_features' : x,
               'train_input_response' : y}
    
    builder = StaticBuilder("MyModel")
    enc_dirs = {'num_layers' : 2,
                'num_nodes' : 128,
                'activation' : 'leaky_relu',
                'net_grow_rate' : 1.0 }

    in0 = builder.addInput(input_dim, name='features')
    
    cust = builder.createCustomNode(1, 1, name="Custom")
    cust_in1 = cust.addInner(3, directives=enc_dirs)
    cust_in2 = cust.addInner(1, directives=enc_dirs)
    cust.addDirectedLink(cust_in1, cust_in2)
    cust.addInput(islot=0, inode_name=cust_in1, inode_islot=0)
    cust.addOutput(oslot=0, inode_name=cust_in2, inode_oslot=0)
    cust.commit()
    
    out0 = builder.addOutput(name='prediction')
    
    builder.addDirectedLink(in0, cust)
    builder.addDirectedLink(cust, out0)
    
    # Pass the builder object to the Regression Model
    reg = Regression(input_dim=2, output_dim=1, builder=builder)
    reg.build()
    reg.visualize_model_graph("two_encoders.png")
    reg.train(dataset, num_epochs=50)

    
if __name__ == '__main__':
  unittest.main(failfast=True)