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
import abc
from collections import namedtuple

import numpy as np
DistPars = namedtuple("DistPars", ("distribution", "get_out_dims"))

import tensorflow as tf
# TODO: Import all these!


from neurolib.encoder import *
# TODO: import all distributions, provide support for all of them

class EncoderNode(abc.ABC):
  """
  The EncoderNode is the basic building block of the neurolib. It corresponds to
  the abstraction of an operation, performed on a set of inputs that results in
  a set of outputs. EncoderNodes can come already built as tensorflow graphs, in
  which case they are a black box with input and output edges, or they can be
  Unbuilt, in which case, the method _build() handles  
  
  Models are directed graphs of Encoders. A Model is itself an EncoderNode so it is
  possible to stitch Models together to produce more complicated ones.
  """
  def __init__(self, label):
    """
    TODO: The client should be able to pass a tensorflow graph directly. In that
    case, EncoderNode should act as a simple wrapper that returns the input and the
    output.
    """
    self.label = label
  
    self.num_inputs = 0
    self.inputs = {}
    self.outputs = {}
    
    self.islot_to_shape = {}
    self.oslot_to_shape = {}
    
    self._is_built = False
    
  def get_inputs(self):
    """
    """
    if not self._is_built:
      raise NotImplementedError("A Node must be built before its inputs and outputs can be "
                                "accessed")
    return self.inputs
    
  def get_outputs(self):
    """
    """
    if not self._is_built:
      raise NotImplementedError("A Node must be built before its inputs and outputs can be "
                                "accessed")
    return self.outputs
  

class UnboundEncoderNode(EncoderNode):
  """
  Classes inheriting from the abstract UnboundEncoderNode, specify tensorflow
  graphs that haven't been built yet. They must implement the _build() method.
  These are the classes that are used to build custom models through a builder
  object
  """
  def __init__(self, label):
    """
    """
    # Important dictionaries for building the Encoder graph 
    self._visited_parents = {}
    self.child_to_oslot = {}
    self.parent_to_islot = {}

    super(UnboundEncoderNode, self).__init__(label)
  
  @abc.abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("Please implement me.")


class InputNode(UnboundEncoderNode):
  """
  An InputNode represents a source in the Encoder graph. It is a node without
  any inputs and with A SINGLE output.
  
  InputNodes are mapped to tensorflow's placeholders.
  """
  def __init__(self, label, output_shape, name=None, batch_size=1, directives={}):
    """
    Handling of the inputs or outputs in an Encoder depends on 2 dictionaries
    that work together. To take the inputs for definiteness, the self.inputs and
    the self.input_to dicts are defined. These dictionaries are initialized
    empty and are filled as Links are added during the specification stage.
    """
    self.name = "In_" + str(label) if name is None else name
    super(InputNode, self).__init__(label)
    self.directives = directives
    
    if isinstance(output_shape, int):
      output_shape = [[output_shape]]
    self.output_shape = output_shape
    self.batch_size = batch_size
    
    self.num_outputs = 1
    self.oslot_to_shape[0] = output_shape
        
  def _build(self, inputs=None):
    """
    """
    if inputs is not None:
      raise ValueError("inuts must be None for an InputNode")

    out_shape = [self.batch_size] + self.output_shape[0]
    print("out_shape:", out_shape)
    name = self.name
    self.outputs[0] = tf.placeholder(tf.float32, shape=out_shape, name=name)
    
    self._is_built = True 


class InnerNode(UnboundEncoderNode):
  """
  An Inner Node represents an encoding in the Encoder graph. It can have an
  arbitrary number of inputs and an arbitrary number of outputs. Its outputs can
  be Deterministic or random samples from a probability distribution. An inner
  node is meant to represent a change of codes.
  
  A typical example of an InnerNode would be neural network. 
  """
  def __init__(self, label, output_shapes, name=None):
    """
    """
    self.name = "Enc_" + str(label) if name is None else name
    super(InnerNode, self).__init__(label)
    
    if isinstance(output_shapes, int):
      output_shapes = [[output_shapes]]
    self.output_shapes = output_shapes
    self.num_outputs = len(output_shapes)
    
    for j in range(self.num_outputs):
      self.oslot_to_shape[j] = output_shapes[j]
      
  @abc.abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("")
  

class OutputNode(UnboundEncoderNode):
  """
  An output node represents a sink in the Encoder graph. Output nodes are
  relatively boring. Their _build() method is trivial since there is nothing
  left to do. It is however important to keep track of them since they are
  obviously needed to stitch Encoders together. They also provide important
  termination conditions for the BFS algorithm that builds the Encoder Graph.
  """
  def __init__(self, label, name=None, directives={}):
    """
    """
    self.name = "Out_" + str(label) if name is None else name
    super(OutputNode, self).__init__(label)
    self.directives = directives
    
  def _build(self):
    """
    Nothing needs to be done for OutputNodes. self.inputs has been already
    updated in the BFS algorithm
    """
    self.inputs[0] = tf.identity(self.inputs[0], name=self.name)
    
    self._is_built = True
            




