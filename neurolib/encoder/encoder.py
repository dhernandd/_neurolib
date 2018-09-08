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
  The EncoderNode is the basic building block of the neurolib. It corresponds to the
  abstraction of an operation, performed on a set of inputs that results in a
  set of outputs
  
  Models are directed graphs of Encoders. A Model is itself an EncoderNode so it is
  possible to stitch Models together to produce more complicated ones.
  """
  def __init__(self, label, directives):
    """
    TODO: The client should be able to pass a tensorflow graph directly. In that
    case, EncoderNode should act as a simple wrapper that returns the input and the
    output.
    """
    self.label = label
    self.directives = directives
    self._visited_parents = {}
  
    self.num_inputs = 0
    self.inputs = {}
    self.outputs = {}

    self.islot_to_shape = {}
    self.oslot_to_shape = {}
    self.child_to_oslot = {}
    self.parent_to_islot = {}
  
  @abc.abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("Please implement me.")
  
  def get_inputs(self):
    """
    """
    return self.outputs
    
  def get_outputs(self):
    """
    """
    return self.outputs
  

class InputNode(EncoderNode):
  """
  """
  def __init__(self, label, output_shape, batch_size=1, directives={}):
    """
    Handling of the inputs or outputs in an Encoder depends on 2 dictionaries
    that work together. To take the inputs for definiteness, the self.inputs and
    the self.input_to dicts are defined. These dictionaries are initialized
    empty and are filled as Links are added during the specification stage.
    """
    self.label = label
    super(InputNode, self).__init__(label, directives)
    
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
    name = 'input_' + str(self.label)
    self.outputs[0] = tf.placeholder(tf.float32, shape=out_shape, name=name) 


class InnerNode(EncoderNode):
  """
  """
  def __init__(self, label, output_shapes, directives={}):
    """
    """
    super(InnerNode, self).__init__(label, directives)
    
    self.output_shapes = output_shapes
    self.num_outputs = len(output_shapes)
    
    for j in range(self.num_outputs):
      self.oslot_to_shape[j] = output_shapes[j]
      
#     # Define slots for dealing with the case of multiple inputs/outputs
#     if "input_slots" in dirs:
#       self.input_slots = dirs["input_slots"]
#     else:
#       self.input_slots = { i : shape for i, shape in enumerate(self.input_shapes) }
#     if "output_slots" in dirs:
#       self.output_slots = dirs["output_slots"]
#     else:
#       self.output_slots = { i : shape for i, shape in enumerate(self.output_shapes) }

  @abc.abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("")
  

class OutputNode(EncoderNode):
  """
  """
  def __init__(self, label, directives={}):
    """
    """
    super(OutputNode, self).__init__(label, directives)
    
  def _build(self):
    """
    """
    pass
        

class DirectedLink():
  """
  """
  def __init__(self, enc1, enc2, directives):
    """
    """
    self.enc1 = enc1
    self.enc2 = enc2
    self.dirs = directives
    self.label = (enc1.label, enc2.label)
    
    # Try automatic pairing of input and output slots. Automatic pairing
    # succeeds if there is exactly one output in enc1 compatible with exactly
    # one input in enc2. Otherwise, request the user to provide the right slots
    try:
      self.paired_slots = self.dirs['paired_slots']
      if not self._are_compatible_shapes(enc1.output_slots[self.paired_slots[0]],
                                         enc2.input_slots[self.paired_slots[1]]):
        raise ValueError("The linked input and output shapes are not "
                         "compatible (DirectLink ", self.label, ")")
    except:
      possible_pairings = [(key1, key2) for key1, shape1 in enc1.output_slots
                           for key2, shape2 in enc2.input_slots if 
                           self._are_compatible_shapes(shape1, shape2)]
      if len(possible_pairings) > 1:
        raise ValueError("Automatic pairing failure. There is more than one "
                         "possible pairing of input and output shapes. Please "
                         "specify the DirectedLink directive `paired_slots`")
      else:
        self.paired_slots = possible_pairings[0]
  
  @staticmethod
  def _are_compatible_shapes(shape1, shape2):
    """
    """
    # For the moment, simoply check if the two shapes are equal
    return shape1 == shape2



