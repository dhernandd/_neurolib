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



class Builder(abc.ABC):
  """
  """
  def __init__(self):
    """
    """
    self.encoder_nodes = {}
    self.input_nodes = {}
    self.output_nodes = {}
    self.num_encoder_nodes = 0
    
  @abc.abstractmethod
  def addInput(self, output_shape, directives={}):
    """
    """
    raise NotImplementedError("Builders must implement addInput")

  @abc.abstractmethod
  def addInner(self, output_shapes, node_class=None,
               directives={}):
    """
    Adds an InnerNode to the Encoder Graph
    """
    raise NotImplementedError("Builders must implement addInner")

  @abc.abstractmethod
  def addOutput(self, directives={}):
    """
    """
    raise NotImplementedError("Builders must implement addInner")
  
  
  