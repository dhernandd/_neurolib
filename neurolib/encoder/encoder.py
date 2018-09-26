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
from abc import abstractmethod
from collections import namedtuple

DistPars = namedtuple("DistPars", ("distribution", "get_out_dims"))

from neurolib.encoder import *

class ANode(abc.ABC):
  """
  An abstract class for Nodes, the basic building block of the neurolib 
  
  An ANode corresponds is an abstraction of an operation, much like
  tensorflow ops, with tensors entering and exiting the node. As opposed to
  tensorflow nodes, Nodes are meant to only represent high level operation,
  each broadly corresponding to an encoding of some input information into some
  output. Some ANodes, such as the Concat ANode also serve in a glue role,
  stitching together nodes of the computational graph.
  
  An ANode specifies a tensorflow graph that has not been
  built yet. A Builder object stacks these Nodes in order to build custom
  models. Descendants of this class must implement the _build() method.
  
  The algorithm to build the tensorflow graph  of the Model depends on 3
  dictionaries that work together: 
  
  self._visited_parents : Keeps track of which among the parents of this node
  have been built. A node can only be built once all of its parents have been
  built
  
  self._child_to_oslot : The keys are the labels of self's children. For each
  key, the only value value is an integer, the oslot in self that maps to that
  child.
  
  self._parent_to_islot : The keys are the labels of self's parents, the only
  value is an integer, the islot in self that maps to that child. 
  """
  def __init__(self, label):
    """
    TODO: The client should be able to pass a tensorflow graph directly. In that
    case, ANode should act as a simple wrapper that returns the input and the
    output.
    """
    self.label = label
  
    self._num_declared_inputs = 0
    self._num_declared_outputs = 0
    
    # Dictionaries for access    
    self._islot_to_itensor = {}
    self._islot_to_shape = {}
    self._oslot_to_otensor = {}
    self._oslot_to_shape = {}
    
    self._visited_parents = {}
    self._child_to_oslot = {}
    self._parent_to_islot = {}
    
    self._is_built = False
    
  @property
  def num_inputs(self):
    """
    """
    return self._num_declared_inputs
  
  @num_inputs.setter
  @abstractmethod
  def num_inputs(self, value):
    """
    """
    raise NotImplementedError("Please implement me")
  
  @property
  def num_outputs(self):
    """
    """
    return self._num_declared_outputs
  
  @num_outputs.setter
  @abstractmethod
  def num_outputs(self, value):
    """
    """
    raise NotImplementedError("Please implement me")
  
  def get_inputs(self):
    """
    """
    if not self._is_built:
      raise NotImplementedError("A Node must be built before its inputs and "
                                "outputs can be accessed")
    return self._islot_to_itensor
    
  def get_outputs(self):
    """
    """
    if not self._is_built:
      raise NotImplementedError("A Node must be built before its inputs and "
                                "outputs can be accessed")
    return self._oslot_to_otensor
  
  def get_output_shapes(self):
    """
    """
    return self._oslot_to_shape
  
  @abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("Please implement me.")
  
  def get_name(self):
    """
    """
    return self.name

  def get_label(self):
    """
    """
    return self.label
  
  




