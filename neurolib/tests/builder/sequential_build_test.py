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

from neurolib.builders.sequential_builder import SequentialBuilder
from neurolib.encoder.input import NormalInputNode

# pylint: disable=bad-indentation, no-member, protected-access

class SequentialModelBuilderTest(tf.test.TestCase):
  """
  """
  def setUp(self):
    """
    """
    tf.reset_default_graph()
    
  @unittest.skipIf(False, "Skipping")
  def test_DeclareModel0(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    builder.addInputSequence(10, 10)

  @unittest.skipIf(False, "Skipping")
  def test_DeclareModel1(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    i1 = builder.addInput(2, iclass=NormalInputNode)
    builder.addInputSequence(10, 10)
    builder.addEvolutionSequence(2, init_states=[i1], num_islots=2)

  @unittest.skipIf(False, "Skipping")
  def test_DeclareModel2(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    i1 = builder.addInput(2, iclass=NormalInputNode)
    builder.addInputSequence(10, 10)
    builder.addEvolutionSequence(2, init_states=[i1], num_islots=2)
    builder.addOutputSequence()

  @unittest.skipIf(False, "Skipping")
  def test_DeclareModel3(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    i1 = builder.addInput(2, iclass=NormalInputNode)
    in1 = builder.addInputSequence(10, 10)
    enc1 = builder.addEvolutionSequence(2, init_states=[i1], num_islots=2)
    o1 = builder.addOutputSequence()
    
    builder.addDirectedLink(in1, enc1, islot=1)
    builder.addDirectedLink(enc1, o1)

  @unittest.skipIf(False, "Skipping")
  def test_BuildModel1(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    i1 = builder.addInput(2, iclass=NormalInputNode)
    in1 = builder.addInputSequence(10, 10)
    enc1 = builder.addEvolutionSequence(2, init_states=[i1], num_islots=2)
    o1 = builder.addOutputSequence()
    
    builder.addDirectedLink(in1, enc1, islot=1)
    builder.addDirectedLink(enc1, o1)
    
    builder.build()

  @unittest.skipIf(False, "Skipping")
  def test_BuildModel2(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    i1 = builder.addInput(2, iclass=NormalInputNode)
    i2 = builder.addInput(4, iclass=NormalInputNode)
    in1 = builder.addInputSequence(10, 10)
    enc1 = builder.addEvolutionSequence(2, init_states=[i1], num_islots=2)
    enc2 = builder.addEvolutionSequence(4, init_states=[i2], num_islots=2)
    o1 = builder.addOutputSequence()
    
    builder.addDirectedLink(in1, enc1, islot=1)
    builder.addDirectedLink(enc1, enc2, islot=1)
    builder.addDirectedLink(enc2, o1)
    
    builder.build()
    
  @unittest.skipIf(False, "Skipping")
  def test_BuildModel3(self):
    """
    """
    builder = SequentialBuilder(max_steps=10, scope="Basic")
    
    i1 = builder.addInput(2, iclass=NormalInputNode)
    i2 = builder.addInput(2, iclass=NormalInputNode)
    
    in1 = builder.addInputSequence(10, 10)
    enc1 = builder.addEvolutionSequence(2,
                                        init_states=[i1,i2],
                                        num_islots=3,
                                        node_class='lstm')
    enc2 = builder.addInnerSequence(4, num_islots=2)
    o1 = builder.addOutputSequence()
    
    builder.addDirectedLink(in1, enc1, islot=2)
    builder.addDirectedLink(in1, enc2)
    builder.addDirectedLink(enc1, enc2, islot=1)
    builder.addDirectedLink(enc2, o1)
    
    builder.build()
    
    e2 = builder.nodes[enc2]
    self.assertEqual(e2._oslot_to_otensor[0].shape.as_list()[-1],
                     4, "Error")
    print("e2._islot_to_itensor", e2._islot_to_itensor)

    
#   @unittest.skipIf(False, "Skipping")
#   def test_DeclareModel10(self):
#     """
#     """
#     builder = SequentialBuilder(max_steps=10, scope="Basic")
#     in1 = builder.addInput(2, iclass=NormalInputNode)
#     i1 = builder.addInputSequence(10)
#     
#     rnn1 = builder.declareRNN()
#     e1 = builder.addEvolutionSequence(3, [in1], mode='forward')
#     rnn1.commit()
#     
#     o1 = builder.addOutputSequence(10)
#     
#     builder.addDirectedLink(i1, e1)
#     builder.addDirectedLink(e1, o1)
#     builder._build()
# 
#   @unittest.skipIf(False, "Skipping")
#   def test_DeclareModel11(self):
#     """
#     """
#     builder = SequentialBuilder(max_steps=10, scope="Basic")
# #     in1 = builder.addInput(2)
#     in2 = builder.addInput(3)
#     in3 = builder.addInput(3)
#     in4 = builder.addInput(3)
#     in5 = builder.addInput(3)
# #     in6 = builder.addInput(3)
#     i1 = builder.addInputSequence(10)
#     
#     rnn1 = builder.declareRNN()
# #     e1 = rnn1.addEvolutionSequence(2, [in1], mode='backward')
#     e1 = rnn1.addEvolutionSequence(3, [in2], mode='forward')
#     e2 = rnn1.addEvolutionSequence(3, [in5], mode='forward')
#     rnn1.addDirectedLink(e1, e2, init_istate=[in3], islot=1)
#     rnn1.addDirectedLink(e2, e1, mode='future', init_istate=[in4], islot=1)
# #     builder.addDirectedLink(e2, e3, mode='future', init_istate=[in6])
#     rnn1.commit()
#     
#     o1 = rnn1.addOutputSequence(10)
#     builder.addDirectedLink(i1, e1)
#     builder.addDirectedLink(i1, e2)
#     builder.addDirectedLink(e2, o1)
#     builder._build()
#     
#   @unittest.skipIf(False, "Skipping")
#   def test_BuildModel0(self):
#     """
#     """
#     print("\nTest: Building a Basic Model")
#     builder = SequentialBuilder(max_steps=10, scope="Basic")
#     in1 = builder.addInput(2)
#     in_seq1 = builder.addInputSequence([10])
#     ev_seq1 = builder.addEvolutionSeq([2], init=(0, in1), mode='future',
#                                       islots=3, cell='basic')
#     ev_seq2 = builder.addEvolutionSeq([2], init=(0, in1), mode='past',
#                                       islots=3)
#     seq1 = builder.addSequence([5])
#     out_seq1 = builder.addOutputSequence()
# 
#     builder.addDirectedLink(in_seq1, ev_seq1)
#     builder.addFutureLink(seq1, ev_seq1)
#     builder.addDirectedLink(seq1, out_seq1)
    
if __name__ == "__main__":
  unittest.main(failfast=True)