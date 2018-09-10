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
from collections import defaultdict

import pydot

from neurolib.builders.builder import Builder
from neurolib.encoder.encoder import ( UnboundEncoderNode, InputNode, OutputNode )
from neurolib.encoder.deterministic import DeterministicEncoding


class StaticModelBuilder(Builder):
  """
  """
  def __init__(self):
    """
    """
    super(StaticModelBuilder, self).__init__()
    
    # The Encoder Graph
    self.input_edges = defaultdict(list)
    self.output_edges = defaultdict(list)
    self.edges = {}
    
    # The graph of the model
    self.model_graph = pydot.Dot(graph_type='digraph')
  
  def addInput(self, output_shape, directives={}):
    """
    Adds an InputNode to the Encoder Graph
    """
    label = self.num_encoder_nodes
    in_node = InputNode(label, output_shape, directives=directives)
    
    # Add properties for visualization
    in_node.vis = pydot.Node('In_' + str(self.num_encoder_nodes))
    self.model_graph.add_node(in_node.vis)

    self.input_nodes[label] = self.encoder_nodes[label] = in_node
    
#     self.model_graph.node(, str(self.num_encoder_nodes))
    self.num_encoder_nodes += 1
    return in_node

  def addInner(self, output_shapes, node_class=DeterministicEncoding,
               directives={}):
    """
    Adds an InnerNode to the Encoder Graph
    """
    label = self.num_encoder_nodes
    enc_node = node_class(label, output_shapes, directives=directives)
    
    # Add properties for visualization
    enc_node.vis = pydot.Node('Enc_' + str(self.num_encoder_nodes))
    self.model_graph.add_node(enc_node.vis)

    self.encoder_nodes[label] = enc_node
    
#     self.model_graph.node(str(self.num_encoder_nodes), str(self.num_encoder_nodes))
    self.num_encoder_nodes += 1
    return enc_node
    
  def addOutput(self, directives={}):
    """
    Adds an OutputNode to the Encoder Graph
    """
    label = self.num_encoder_nodes
    out_node = OutputNode(label, directives=directives)
    
    # Add properties for visualization
    out_node.vis = pydot.Node('Out_' + str(self.num_encoder_nodes))
    self.model_graph.add_node(out_node.vis)

    self.output_nodes[label] = self.encoder_nodes[label] = out_node
    
#     self.model_graph.node(str(self.num_encoder_nodes), str(self.num_encoder_nodes))
    self.num_encoder_nodes += 1
    return out_node
    
  def addDirectedLink(self, node1, node2, oslot=None):
    """
    Adds directed links to the corresponding dictionaries. Moreover, it creates
    and updates the adjacency matrix and adjacency list properties of the encoder graph
    """
    # Deal with different item types. The client may provide as arguments,
    # either EncoderNodes or integers. Get the EncoderNodes in the latter case
    if isinstance(node1, int):
      node1 = self.encoder_nodes[node1]
    if isinstance(node2, int):
      node2 = self.encoder_nodes[node2]
    
    nnodes = self.num_encoder_nodes
    if not node1.oslot_to_shape:
      if isinstance(node1, OutputNode):
        raise ValueError("You cannot define outgoing directed links for OutputNodes")
      else:
        raise ValueError("Node1 appears to have no outputs. This software has no "
                         "clue why that would be. Please report to my master.")

    # Initialize the graph representations the first time a DirectedLink is
    # added. Otherwise, add dimensions to the adjacency matrix representation of
    # the Model graph
    if not hasattr(self, "adj_matrix"):
      self.adj_matrix = [[0]*nnodes for _ in range(nnodes)]
      self.adj_list = [[] for _ in range(nnodes)]
    else:
      if nnodes > len(self.adj_matrix):
        l = len(self.adj_matrix)
        for row in range(l):
          self.adj_matrix[row].append([0]*(nnodes-l))
        for row in range(nnodes-l):
          self.adj_matrix.append([0]*nnodes)
    
    # Update the representations of the Model graph with this link. Also 
    if isinstance(node1, UnboundEncoderNode) and isinstance(node2, UnboundEncoderNode):
      self._check_items_do_exist()
      self.adj_matrix[node1.label][node2.label] = 1
      self.adj_list[node1.label].append(node2.label)
      
      self.model_graph.add_edge(pydot.Edge(node1.vis, node2.vis))
#       self.model_graph.edge(str(node1.label), str(node2.label))
    else:
      raise ValueError("The endpoints of the links must be either Encoders or "
                       "integers labeling Encoders")
      
    # Update the encoder properties, input_to, output_to AND crucially,
    # self.input_shapes for the out nodes
    if len(node1.oslot_to_shape) > 1:
      if oslot is None:
        raise ValueError("The in-node has more than one output, so assignment to "
                         "the out-node is ambiguous. You must specify the output "
                         "slot. The current output slots for node 1 are: ",
                         node1.oslot_to_shape)
    else:
      oslot = 0
    # Fill the all important dictionaries child_to_oslot and parent_to_islot.
    # For node.child_to_oslot[key] = value, key represents the labels of the
    # children of node, and the values are the indices of the output slot in
    # node that leaves for that child. Analogously, in node.parent_to_islot[key]
    # = value, the keys are the labels of the parents of node and the values are
    # the input slots in node corresponding to each key.
    exchanged_shape = node1.oslot_to_shape[oslot]
    islot = node2.num_inputs
    node2.islot_to_shape[islot] = exchanged_shape
    node1.child_to_oslot[node2.label] = oslot
    node2.parent_to_islot[node1.label] = islot    
    node2.num_inputs += 1     

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
    TODO:
    """
    pass
        
  def build(self):
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
    
    print("Building the model...")
    visited = [False for _ in range(self.num_encoder_nodes)]
    queue = []
    for cur_inode_label in self.input_nodes:
      
      # start BFS from this input
      queue.append(cur_inode_label)
      while queue:
        # A node is visited by definition once it is popped from the queue
        cur_node_label = queue.pop(0)
        visited[cur_node_label] = True
        cur_node = self.encoder_nodes[cur_node_label]

        print("Building node: ", cur_node.label)
        # Build the tensorflow graph for this Encoder
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
          child_node = self.encoder_nodes[child_label]
          child_node._visited_parents[cur_node_label] = True
          
          # Get islot and oslot
          oslot = cur_node.child_to_oslot[child_label]
          islot = child_node.parent_to_islot[cur_node_label]
          
          # Fill the inputs of the child node
          child_node.inputs[islot] = cur_node.get_outputs()[oslot]
          
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
    

      
      
      
    
  
  
  