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

from neurolib.models.deterministic import NeuralNetRegression
from tensorflow.contrib.layers.python.layers import fully_connected

def make_data_iterator(data, batch_size=1, shuffle=True):
    """
    """
    nsamps = len(data[0])
    l_inds = np.arange(nsamps)
    if shuffle: 
        np.random.shuffle(l_inds)
    
    for i in range(0, nsamps, batch_size):
        yield [ d[l_inds[i:i+batch_size]] for d in data ]
        
class DeterministicCodeFullTest(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """  
  @unittest.skip("Skipping")
  def test_init(self):
    """
    """
    tf.reset_default_graph()
    NeuralNetRegression(input_dim=10, output_dim=1)
    
  @unittest.skip("Skipping")
  def test_build(self):
    """
    """
    tf.reset_default_graph()
    dc = NeuralNetRegression(input_dim=10, output_dim=1)
    dc.build()
#     dc.visualize_graph()
    
#   @unittest.skip("Skipping")
  def test_train(self):
    """
    """
    x = 10.0*np.random.randn(100, 2)
    y = x[:,0:1] + 1.5*x[:,1:]# + 3*x[:,1:]**2 + 0.5*np.random.randn(100,1)
    dataset = {'train_features' : x,
               'train_response' : y}
    
    tf.reset_default_graph()
    dc = NeuralNetRegression(input_dim=2, output_dim=1)
    dc.build()
    dc.train(dataset, num_epochs=10)
    

if __name__ == '__main__':
  tf.test.main()