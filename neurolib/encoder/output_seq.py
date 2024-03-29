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

from neurolib.encoder.anode import ANode

# pylint: disable=bad-indentation, no-member, protected-access

class OutputSequence(ANode):
  """
  """
  num_expected_inputs = 1
  num_expected_outputs = 0
  
  def __init__(self, label, name=None):
    """
    Initialize the OutputNode
    
    Args:
      label (int): A unique integer identifier for the node.
      
      name (str): A unique name for this node.
    """
    self.name = "OutSeq_" + str(label) if name is None else name
    self.label = label
    super(OutputSequence, self).__init__()
    
  def _build(self):
    """
    Build the OutputNode.
    
    Specifically, rename the input tensor to the name of the node so that it can
    be easily accessed.
    
    NOTE: The _islot_to_itensor attribute of this node has been updated by the
    Builder object during processing of this OutputNode's parent by the
    Builder build algorithm
    """
    print("\nOutput:", self.name,
          "\nself._islot_to_itensor[0]", type(self._islot_to_itensor[0]) )
    self._islot_to_itensor[0] = tf.identity(self._islot_to_itensor[0],
                                            name=self.name)
    
    self._is_built = True