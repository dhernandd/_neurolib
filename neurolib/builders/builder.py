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

import pydot

from neurolib.encoder.deterministic import DeterministicNNNode

def check_name(f):
  def f_checked(obj, *args, **kwargs):
    if 'name' in kwargs:
      if kwargs['name'] in obj.nodes:
        raise AttributeError("The name", kwargs["name"], "already corresponds "
                             "to a node in this graph")
    return f(obj, *args, **kwargs)
  
  return f_checked


class Builder(abc.ABC):
  """
  An abstract class for building graphical models of Encoder nodes.
  """
  def __init__(self, scope, batch_size=1):
    """
    """
    self.scope = scope
    self.batch_size = batch_size
    
    self.num_nodes = 0
    
    # Dictionaries that map name/label to node for the three node types.
    self.nodes = {}
    self.input_nodes = {}
    self.output_nodes = {}
    self._label_to_node = {}

    # The graph of the model
    self.model_graph = pydot.Dot(graph_type='digraph')

  @check_name
  def addInner(self, *main_params, node_class=DeterministicNNNode, name=None,
               **dirs):
    """
    Adds an InnerNode to the Encoder Graph
    """
    label = self.num_nodes
    self.num_nodes += 1
    
    if node_class._requires_builder:
      enc_node = node_class(label, *main_params, name=name,
                            builder=self, 
                            batch_size=self.batch_size,
                            **dirs)
    else:
      enc_node = node_class(label, *main_params, name=name,
                            batch_size=self.batch_size,
                            **dirs)
      
    self.nodes[enc_node.name] = self._label_to_node[label] = enc_node
  
    # Add properties for visualization
    self.model_graph.add_node(enc_node.vis)
    
    return enc_node.name
  
  @abc.abstractmethod
  def _build(self): 
    """
    """
    raise NotImplementedError("Builders must implement build")
  
  def visualize_model_graph(self, filename="model_graph"):
    """
    Generates a representation of the computational graph
    """
    self.model_graph.write_png(self.scope+filename)
  
  
