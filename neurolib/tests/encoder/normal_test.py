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
import tensorflow as tf

from neurolib.encoder.normal import NormalTriLNode
from neurolib.builders.static_builder import StaticModelBuilder

class NormalTriLFullTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()
        
  @unittest.skipIf(False, "Skipping")
  def test_init_build(self):
    """
    """
    print("Test 0: Initialization + Build")
    output_shape = [3]
    
    builder = StaticModelBuilder("NormalTest")
    i0 = builder.addInput(output_shape, name='features', directives={})
    enc0 = builder.addInner([4], name='NormalTril',
                            node_class=NormalTriLNode, directives={})
    o0 = builder.addOutput(name='response', directives={})
    
    builder.addDirectedLink(i0, enc0)
    builder.addDirectedLink(enc0, o0, oslot=0)
    print("Label of this node:", builder.nodes[enc0].label)
    builder._build()
#     builder.visualize_model_graph()
    
    
if __name__ == '__main__':
  tf.test.main()
  
  