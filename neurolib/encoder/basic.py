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
import pydot
import tensorflow as tf

from neurolib.encoder import _globals as dist_dict
from neurolib.encoder.anode import ANode


class OutputNode(ANode):
  """
  An OutputNode represents a sink in the Encoder graph. It is a
  node with exactly 1 input and zero outputs. 
  
  The _build method of an OutputNode does little. An important function is to
  rename the incoming tensor, using tf.identity, to the name required by the
  Trainer object. Another important role of OutputNodes is to provide
  termination conditions for the BFS algorithm that builds the Encoder Graph.
  
  Some InnerNodes, for instance stochastic ones - InnerNodes that represent an
  encoding as a random variable - typically add OutputNodes automatically,
  corresponding to the order statistics of the involved distribution. A Normal
  EncoderNode for instance would automatically addOutputNodes for the mean and
  standard deviation.
  """
  num_expected_inputs = 1
  num_expected_outputs = 0
  
  def __init__(self, label, name=None, directives={}):
    """
    """
    self.name = "Out_" + str(label) if name is None else name
    self.directives = directives
#     self._num_declared_outputs = 0
    
    # Initialize the Encoder dictionaries
    super(OutputNode, self).__init__(label)
    
    # Add visualization
    self.vis = pydot.Node(self.name)

  @ANode.num_inputs.setter
  def num_inputs(self, value):
    """
    """
    if value > self.num_expected_inputs:
      raise AttributeError("Attribute num_inputs of OutputNodes must be either 0 "
                           "or 1")
    self._num_declared_inputs = value

  @ANode.num_outputs.setter
  def num_outputs(self, value):
    """
    """
    raise AttributeError("Assignment to attribute num_outputs of OutputNodes is "
                         " disallowed. num_outputs is set to 0 for an OutputNode")
    
  def _build(self):
    """
    Builds the tensorflow ops corresponding to this OutputNode.
    
    A) Rename the input tensor
    
    NOTE: The _islot_to_itensor attribute of this node has been updated during
    the processing of the parent of this OutputNode in the BFS algorithm.
    """
    # Stage A
    self._islot_to_itensor[0] = tf.identity(self._islot_to_itensor[0],
                                            name=self.name)
    
    self._is_built = True
    

class EncoderNode(ANode):
  """
  TODO: CHANGE THE NAME, CloneNode is also "inner"
  
  An EncoderNode represents an encoding in the Encoder graph. It can have an
  arbitrary number of inputs and an arbitrary number of outputs. Its outputs can
  be Deterministic or random samples from a probability distribution. An inner
  node is meant to represent a change of codes.  
  """
  def __init__(self, label, main_output_shapes):
    """
    """
    super(EncoderNode, self).__init__(label)
    
    # TODO: Remove this, it is useless
    if isinstance(main_output_shapes, int):
      main_output_shapes = [[main_output_shapes]]
    self.main_output_shapes = main_output_shapes
          
    # Add visualization
    self.vis = pydot.Node(self.name, shape='box')
      

class MergeConcatNode(ANode):
  """
  """
  num_expected_outputs = 1
  
  def __init__(self, label, num_mergers=2, axis=1, name=None,
               builder=None, directives={}):
    """
    """
    self.label = label
    super(MergeConcatNode, self).__init__(label)
    self.builder = builder
    self.directives = directives
    
    self.num_expected_inputs = self.num_mergers = num_mergers
    self.axis = axis
    self.name = "Concat_" + str(label) if name is None else name
    
    self._num_declared_outputs = 1
    # Add visualization
    self.vis = pydot.Node(self.name, shape='box')

    
  @ANode.num_inputs.setter
  def num_inputs(self, value):
    """
    """
    if value > self.num_expected_inputs:
      raise AttributeError("Attribute num_inputs of this MergeConcatNode must be "
                           "at most", self.num_expected_inputs)
    self._num_declared_inputs = value

  @ANode.num_outputs.setter
  def num_outputs(self, value):
    """
    """
    if value > self.num_expected_outputs:
      raise AttributeError("Attribute num_outputs of this CloneNode must be "
                           "at most", self.num_expected_outputs)
    self._num_declared_outputs = value
    
  def _build(self):
    """
    """
    values = list(self._islot_to_itensor.values())
    self._oslot_to_otensor[0] = tf.concat(values, axis=self.axis)
    
    self._is_built = True
    
  def _update_when_linked_as_node2(self):
    """
    """
    if self.num_inputs == self.num_expected_inputs:
      s = 0
      oshape = list(self._islot_to_shape[0])
      print('oshape concat:', oshape)
      for islot in range(self.num_expected_inputs):
        print('islot, shape', islot, self._islot_to_shape[islot])
        s += self._islot_to_shape[islot][self.axis]
        print('s', s)
      oshape[self.axis] = s
      print('final oshape concat:', oshape)
    
      self._oslot_to_shape[0] = oshape
    
    
class CloneNode(ANode):
  """
  """
  num_expected_inputs = 1
  
  def __init__(self, label, num_clones=2, name=None, builder=None, directives={}):
    """
    """
    self.label = label
    super(CloneNode, self).__init__(label)
    
    self.directives = directives
    self.builder = builder
    self.num_clones = self.num_expected_outputs = num_clones
    
    self.name = "Clone_" + str(label) if name is None else name

    # Add visualization
    self.vis = pydot.Node(self.name, shape='box')
    
  def _build(self):
    """
    """
    x_in = self._islot_to_itensor[0]
    for _ in range(self.num_clones):
      name = ( x_in.name + '_clone_' + str(self.num_outputs) if self.name is None
               else self.name )
      self._oslot_to_otensor[self.num_outputs] = tf.identity(x_in, name)
      self.num_outputs += 1
    
    self._is_built = True

  @ANode.num_inputs.setter
  def num_inputs(self, value):
    """
    """
    if value > self.num_expected_inputs:
      raise AttributeError("Attribute num_inputs of CloneNodes must be either 0 "
                           "or 1")
    self._num_declared_inputs = value

  @ANode.num_outputs.setter
  def num_outputs(self, value):
    """
    """
    if value > self.num_expected_outputs:
      raise AttributeError("Attribute num_outputs of this CloneNode must be "
                           "at most", self.num_expected_outputs)
    self._num_declared_outputs = value
    
  def _update_when_linked_as_node2(self):
    """
    """
    for oslot in range(self.num_expected_outputs):
      self._oslot_to_shape[oslot] = self._islot_to_shape[0]
    

if __name__ == '__main__':
  print(dist_dict)
    