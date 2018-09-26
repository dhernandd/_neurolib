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
import unittest
import tensorflow as tf

from neurolib.encoder.basic import InputNode, OutputNode

toRun = [0, 1, 2]

class BasicNodeFullTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()
    
  @unittest.skipIf(0 not in toRun, "Skipping")
  def test_in_init(self):
    """
    """
    i_node = InputNode(0, [10])
    name = i_node.get_name()
    label = i_node.get_label()
    self.assertEqual(label, 0, "Label assignment failure")
    print("This node name: ", name)
    
  @unittest.skipIf(1 not in toRun, "Skipping")
  def test_in_build(self):
    """
    """
    i_node = InputNode(0, [10])
    i_node._build()
    Y = i_node.get_outputs()
    print("The outputs are: ", Y)
    
  @unittest.skipIf(2 not in toRun, "Skipping")
  def test_out_init(self):
    """
    """
    o_node = OutputNode(0)
    name = o_node.get_name()
    label = o_node.get_label()
    self.assertEqual(label, 0, "Label assignment failure")
    print("This node name: ", name)
        
  
if __name__ == '__main__':
  tf.test.main()
  
  