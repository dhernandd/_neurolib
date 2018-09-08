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

from neurolib.builders.builder import StaticModelBuilder

class StaticModelBuilderBasicTest(tf.test.TestCase):
  """
  """
  @unittest.skip
  def test_init(self):
    """
    """
    builder = StaticModelBuilder()
    builder.addInput(10)
    print('Nodes in builder:', builder.input_nodes)
    
  @unittest.skip
  def test_addInner(self):
    """
    """
    tf.reset_default_graph()
    builder = StaticModelBuilder()
    builder.addInput(10)
    builder.addInner([[3], [4]])
    print('Nodes in builder:', builder.encoder_nodes)

  @unittest.skip
  def test_addOutput(self):
    """
    """
    tf.reset_default_graph()
    builder = StaticModelBuilder()
    builder.addInput(10)
    builder.addInner([[3], [4]])
    builder.addOutput()
    print('Nodes in builder:', builder.encoder_nodes)
  
  @unittest.skip
  def test_addDirectedLinks(self):
    """
    Builds the simplest model possible, check that it is built correctly.
    """
    tf.reset_default_graph()
    builder = StaticModelBuilder()
    in1 = builder.addInput(10)
    enc1 = builder.addInner([[3]])
    out1 = builder.addOutput()
    builder.addDirectedLink(in1, enc1)
    builder.addDirectedLink(enc1, out1)
    builder.build()

  @unittest.skip
  def test_BuildModel1(self):
    """
    Builds a model with 2 outputs... OK!
    """
    tf.reset_default_graph()
    builder = StaticModelBuilder()
    in1 = builder.addInput(10)
    enc1 = builder.addInner([[3], [4]])
    out1 = builder.addOutput()
    out2 = builder.addOutput()
    builder.addDirectedLink(in1, enc1)
    builder.addDirectedLink(enc1, out1, oslot=0)
    builder.addDirectedLink(enc1, out2, oslot=1)
    builder.build()

  @unittest.skip
  def test_BuildModel2(self):
    """
    Builds a model with 2 inputs
    """
    tf.reset_default_graph()
    builder = StaticModelBuilder()
    in1 = builder.addInput(10)
    in2 = builder.addInput(20)
    enc1 = builder.addInner([[3]])
    out1 = builder.addOutput()
    builder.addDirectedLink(in1, enc1)
    builder.addDirectedLink(in2, enc1,)
    builder.addDirectedLink(enc1, out1)
    builder.build()
    
  def test_BuildModel3(self):
    """
    Try to break it the algorithm... !!! Guess not mdrfkr.
    """
    tf.reset_default_graph()
    builder = StaticModelBuilder()
    in1 = builder.addInput(10)
    in2 = builder.addInput(20)
    enc1 = builder.addInner([[3], [5]])
    enc2 = builder.addInner([[4]])
    out1 = builder.addOutput()
    out2 = builder.addOutput()
    builder.addDirectedLink(in1, enc1)
    builder.addDirectedLink(in2, enc2,)
    builder.addDirectedLink(enc1, enc2, oslot=0)
    builder.addDirectedLink(enc1, out1, oslot=1)
    builder.addDirectedLink(enc2, out2)
    builder.build()

    
if __name__ == "__main__":
  tf.test.main()