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

from neurolib.builders.static_builder import StaticModelBuilder
from neurolib.encoder.normal import NormalTriLNode

class CustomEncoderNormalTest(tf.test.TestCase):
  """
  """
#   @unittest.skip
  def test_add_encoder(self):
    """
    Add an Encoder Node to the Custom Encoder
    """
    tf.reset_default_graph()
    builder = StaticModelBuilder("MyModel")
    in1 = builder.addInput(10)

    cust = builder.createCustomNode("Custom")
    cust_in1 = cust.addInner(3, node_class=NormalTriLNode)
    cust.commit()
     
    o1 = builder.addOutput()
    builder.addDirectedLink(in1, cust)
    builder.addDirectedLink(cust, o1, oslot=0)
     
    builder._build()
    
    
    
    

if __name__ == "__main__":
  tf.test.main()