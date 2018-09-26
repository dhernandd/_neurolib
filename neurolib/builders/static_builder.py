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
from abc import abstractmethod

import pydot
import tensorflow as tf

from neurolib.builders.builder import Builder
from neurolib.encoder.basic import InputNode, OutputNode
from neurolib.encoder.deterministic import DeterministicNode
from neurolib.encoder.normal import NormalTriLNode
from neurolib.encoder.encoder import ANode
from neurolib.encoder.custom import CustomEncoderNode

bayesian_nodes = [NormalTriLNode]


def check_name(f):
  def f_checked(obj, *args, **kwargs):
#     print("'name' in kwargs", 'name' in kwargs)
    if 'name' in kwargs:
      print("type(obj)", type(obj))
      print("kwargs['name'] in obj.nodes", kwargs['name'], 
            kwargs['name'] in obj.nodes)
      if kwargs['name'] in obj.nodes:
        raise AttributeError("The name", kwargs["name"], "already corresponds "
                             "to a node in this graph")
    return f(obj, *args, **kwargs)
  
  return f_checked


class ModelBuilder(Builder):
  """
  """
  def __init__(self, scope, batch_size=1):
    """
    """
    super(ModelBuilder, self).__init__(scope, batch_size=batch_size)
  
  @abstractmethod
  def addInput(self, output_shape, name=None, directives={}):
    """
    Adds an InputNode to the Encoder Graph. ModelBuilders must implement this
    """
    raise NotImplementedError("ModelBuilders must implement addInput")

  @abstractmethod
  def addOutput(self, name=None, directives={}):
    """
    Adds an OutputNode to the Encoder Graph. ModelBuilders must implement this
    """
    raise NotImplementedError("ModelBuilders must implement addOutput")


class StaticModelBuilder(ModelBuilder):
  """
  A class for building StaticModels (Models that do not involve sequential
  data).
  """
  def __init__(self, scope=None, batch_size=1):
    """
    """
    super(StaticModelBuilder, self).__init__(scope, batch_size=batch_size)
            
    # The pydot graph of the model
    self.model_graph = pydot.Dot(graph_type='digraph')
  
  @check_name
  def addInput(self, output_shape, name=None, directives={}):
    """
    Adds an InputNode to the Encoder Graph
    """
    label = self.num_nodes
    self.num_nodes += 1
    in_node = InputNode(label, output_shape, batch_size=self.batch_size,
                        name=name, directives=directives)
#     self.input_nodes[label] = self.nodes[label] = in_node
    name = in_node.name
    self.input_nodes[name] = self.nodes[name] = self._label_to_node[label] = in_node
    
    # Add properties for visualization
    self.model_graph.add_node(in_node.vis)

#     return in_node.label
    return name
    
  @check_name
  def addOutput(self, name=None, directives={}):
    """
    Adds an OutputNode to the Encoder Graph
    """
    label = self.num_nodes
    self.num_nodes += 1
    out_node = OutputNode(label, name=name, directives=directives)
#     self.output_nodes[label] = self.nodes[label] = out_node
    name = out_node.name
    self.output_nodes[name] = self.nodes[name] = self._label_to_node[label] = out_node
    
    # Add properties for visualization
    self.model_graph.add_node(out_node.vis)

#     return out_node.label
    return name
    
  def addDirectedLink(self, node1, node2, oslot=None, islot=None):
    """
    Adds directed links to the Encoder graph. The method functions in several
    stages detailed below:
    
    A) Deal with different item types. The client may provide as arguments,
    either EncoderNodes or integers. Get the EncoderNodes in the latter case
 
    B) Check that the provided oslot for node1 is free. Otherwise, raise an
    exception.
    
    C) Initialize/Add dimensions the graph representations stored in the
    builder. Specifically, the first time a DirectedLink is added an adjacency
    matrix and an adjacency list are created. From then on, the appropriate
    number of dimensions are added to these representations.

    D) Update the representations to represent the new link. 
    
    E) Fill the all important dictionaries _child_to_oslot and _parent_to_islot.
    For node._child_to_oslot[key] = value, key represents the labels of the
    children of node, while the values are the indices of the oslot in node
    that outputs to that child. Analogously, in node._parent_to_islot[key] =
    value, the keys are the labels of the parents of node and the values are the
    input slots in node corresponding to each key.
    
    F) Possibly update the attributes of node2. In particular deal with nodes
    whose output shapes are dynamically inferred. This is important for nodes such
    as CloneNode and ConcatNode whose output shapes are not provided at
    creation. Once these nodes gather their inputs, they can infer their
    output_shape at this stage.
    """
    # Stage A
    if isinstance(node1, str):
      node1 = self.nodes[node1]
    if isinstance(node2, str):
      node2 = self.nodes[node2]
    
    # Stage B
    nnodes = self.num_nodes
    if not node1._oslot_to_shape:
      if isinstance(node1, OutputNode):
        raise ValueError("You cannot define outgoing directed links for OutputNodes")
      else:
        raise ValueError("Node1 appears to have no outputs. This software has no "
                         "clue why that would be.\n Please report to my master.")

    # Stage C
    print('Adding dlink', node1.label, ' -> ', node2.label)
    if not hasattr(self, "adj_matrix"):
      self.adj_matrix = [[0]*nnodes for _ in range(nnodes)]
      self.adj_list = [[] for _ in range(nnodes)]
    else:
#       print(self.adj_matrix)
      print('Before:', self.adj_list)
      if nnodes > len(self.adj_matrix):
        l = len(self.adj_matrix)
        for row in range(l):
          self.adj_matrix[row].extend([0]*(nnodes-l))
        for _ in range(nnodes-l):
          self.adj_matrix.append([0]*nnodes)
          self.adj_list.append([])
    
    # Stage D
    if isinstance(node1, ANode) and isinstance(node2, ANode):
#       print(self.adj_matrix)
#       print(self.adj_list)
      self._check_items_do_exist()
      self.adj_matrix[node1.label][node2.label] = 1
      self.adj_list[node1.label].append(node2.label)
#       print(self.adj_matrix)
      print('After:', self.adj_list)
      
      self.model_graph.add_edge(pydot.Edge(node1.vis, node2.vis))
    else:
      raise ValueError("The endpoints of the links must be either Encoders or "
                       "integers labeling Encoders")
      
    # Stage E
    if node1.num_expected_outputs > 1:
      if oslot is None:
        raise ValueError("The in-node has more than one output slot, so pairing "
                         "to the out-node is ambiguous.\n You must specify the "
                         "output slot. The declared output slots for node 1 are: ",
                         node1._oslot_to_shape)
    else:
      oslot = 0
    if node2.num_expected_inputs > 1:
      if islot is None:
        raise ValueError("The out-node has more than one input slot, so pairing "
                         "from the in-node is ambiguous.\n You must specify the " 
                         "input slot")
    else:
      islot = 0
    exchanged_shape = node1._oslot_to_shape[oslot]
    node1._child_to_oslot[node2.label] = oslot
    node1.num_outputs += 1

    print('Exchanged shape:', exchanged_shape)
    if islot in node2._islot_to_shape:
      raise AttributeError("That input slot is already occupied. Assign to "
                           "a different islot")
    node2._islot_to_shape[islot] = exchanged_shape
    node2._parent_to_islot[node1.label] = islot    
    node2.num_inputs += 1
    
    # Stage F
    update = getattr(node2, '_update_when_linked_as_node2', None)
    if callable(update):
      node2._update_when_linked_as_node2()

    # Initialize _visited_parents for the child node. This is used in the build
    # algorithm below.
    node2._visited_parents[node1.label] = False
      
  def _check_items_do_exist(self):
    """
    TODO:
    """
    pass
      
  def check_graph_correctness(self):
    """
    Checks the coding graph outlined so far. 
    
    TODO:
    """
    pass
        
  def createCustomNode(self, name=None):
    """
    """
    label = self.num_nodes
    self.num_nodes += 1

    self.custom_encoders = {}
    custom_builder = StaticModelBuilder(name)
    cust = CustomEncoderNode(label, builder=custom_builder, scope=name)
    self.custom_encoders[name] = self.nodes[label] = cust
        
    return cust
  
  def get_custom_encoder(self, name):
    """
    """
    return self.custom_encoders[name] 
  
  def add_to_custom(self, cust, output_shapes, name=None,
                    node_class=DeterministicNode, directives={}):
    """
    """
    cust.builder.addInner(output_shapes, name=name, node_class=node_class,
                          directives=directives)
    
  def _build(self):
    """
    Builds the model for this builder.
    # put all nodes in a waiting list of nodes
    # for node in input_nodes:
      # start BFS from node. Add node to queue.
      # (*)Dequeue, mark as visited
      # build the tensorflow graph with the new added node
      # Look at all its children nodes.
      # For child in children of node
          Add node to the list of inputs of child
      #   have we visited all the parents of child?
          Yes
            Add to the queue
      # Go back to (*)
      * If the queue is empty, exit, start over from the next input node until all 
      # have been exhausted
      
      # TODO: deal with back links.
    """       
    self.check_graph_correctness()
    
    print('\nBEGIN MAIN BUILD')
    print("Building the model...")
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE): 
      visited = [False for _ in range(self.num_nodes)]
      queue = []
      for cur_inode_name in self.input_nodes:
        cur_inode_label = self.get_label(cur_inode_name)
        
        # start BFS from this input
        queue.append(cur_inode_label)
        while queue:
          # A node is visited by definition once it is popped from the queue
          cur_node_label = queue.pop(0)
          visited[cur_node_label] = True
          cur_node = self._label_to_node[cur_node_label]
  
          print("Building node: ", cur_node.label, cur_node.name)
          # Build the tensorflow graph for this Encoder
          print("_islot_to_itensor", cur_node._islot_to_itensor)
          cur_node._build()
          
          # TODO: If the node is an input or an output, add it to the Model's lists
          if isinstance(cur_node, InputNode):
            pass
  #           self.inputs[input_slots] = (cur_node.get_outputs())
  #           input_slots += 1
          if isinstance(cur_node, OutputNode):
            pass
  #           self.outputs[output_slots] = (cur_node.get_input())
  #           output_slots += 1
          
          # Go over the current node's children
          for child_label in self.adj_list[cur_node_label]:
            child_node = self._label_to_node[child_label]
            child_node._visited_parents[cur_node_label] = True
            
            # Get islot and oslot
            oslot = cur_node._child_to_oslot[child_label]
            islot = child_node._parent_to_islot[cur_node_label]
            
            # Fill the inputs of the child node
            print('cur_node', cur_node_label, cur_node.name)
            print('cur_node.get_outputs()', cur_node.get_outputs() )
            child_node._islot_to_itensor[islot] = cur_node.get_outputs()[oslot]
            if isinstance(child_node, CustomEncoderNode):
              enc, enc_islot = child_node._islot_to_enc_islot[islot]
              enc._islot_to_itensor[enc_islot] = cur_node.get_outputs()[oslot]
            
            # If the child is an OutputNode, we can append to the queue right away
            # (OutputNodes have only one input)
            if isinstance(child_node, OutputNode):
              queue.append(child_node.label)
              continue
            
            # A child only gets added to the queue, i.e. ready to be built, once
            # all its parents have been visited ( and hence, produced the
            # necessary inputs )
            if all(child_node._visited_parents.items()):
              queue.append(child_node.label)
    
    print('Finished building')
    print('END MAIN BUILD')
    

  def get_label(self, name):
    """
    """
    return self.nodes[name].label
  

  
      
    
  
  
  