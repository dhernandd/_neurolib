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

from neurolib.encoder.normal import NormalTrilEncoding

class DeterministicEncoderFullTest(tf.test.TestCase):
  """
  """
  @unittest.skip("Skipping")
  def test_init(self):
    """
    """
    label = 1
    output_shapes = [[3]]
    tf.reset_default_graph()
    det_enc = NormalTrilEncoding(label, output_shapes)
    print("Label of this node:", det_enc.label)
    self.assertEqual(det_enc.label, label, msg="Label is not assigned correctly")
    self.assertEqual(output_shapes[0], det_enc.oslot_to_shape[0], msg="Output shape does not "
                     "match the shape indicated in the oslot")
    
  def test_build(self):
    """
    """
    label = 1
    output_shapes = [[3]]
    tf.reset_default_graph()
    det_enc = NormalTrilEncoding(label, output_shapes)
    det_enc.inputs[0] = tf.placeholder(dtype=tf.float32, shape=[1, 10])
    det_enc._build()
#     self.assertEqual(det_enc.label, label, msg="Label is not assigned correctly")
    
if __name__ == '__main__':
  tf.test.main()