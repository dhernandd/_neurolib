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

from neurolib.encoder.deterministic import DeterministicNode

class DeterministicEncoderFullTest(tf.test.TestCase):
  """
  """  
  def setUp(self):
    """
    """
    tf.reset_default_graph()
  
  @unittest.skipIf(False, "")
  def test0_init(self):
    """
    """
    print("Test 0: Initialization")
    label = 1
    
    det = DeterministicNode(label, [16], name="MyDet")
    self.assertEqual(1, det.label)
    name = det.get_name()
    print("This node's name:", name)

  @unittest.skipIf(False, "")
  def test1_build(self):
    """
    """
    print("Test 1: Build")    
    
    label = 1
    det = DeterministicNode(label, [16], name="MyDet")
    self.assertEqual(1, det.label)
    
    # Manually fill the _islot_to_itensor dict
    X = tf.placeholder(tf.float32, [1, 4], 'my_pl')
    det._islot_to_itensor[0] = X
    
    det._build()
    self.assertEqual(True, det._is_built)
    self.assertEqual(det._oslot_to_otensor[0].shape.as_list()[-1],
                     16, "Shape not defined correctly")
    
    
if __name__ == '__main__':
  tf.test.main()
    