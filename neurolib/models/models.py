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
from builtins import isinstance


class Model():
  """
  """
  def __init__(self):
      """
      """
      pass
  
  def build(self):
    """
    """
    pass
    
  def update(self):
    """
    TODO: Fill
    """
    raise NotImplementedError("")


  def train(self):
    """
    """
    pass
    
    
class Encoder():
  """
  """
  pass


class Unsupervised():
  """
  """
  def __init__(self, Y, t_init=None, specs=None):
    """
    """
    self.Y = Y
    self.ydim = int(Y.shape[1])

    if t_init is not None:
      self._initialize_from_tensors(t_init)
      self.is_initialized = True
    elif specs is not None:
      self._initialize_from_specs(specs)
      self.is_initialized = True
    else:
      self.is_initialized = False
    
  def _initialize_from_tensors(self):
    """
    """
    raise NotImplementedError("")

  def _initialize_from_specs(self, specs):
    """
    TODO: Fill ValueError eception
    """
    nodes_quality = []
    if isinstance(specs, dict):
      pass
    elif isinstance(specs, str):
      encoders_list_specs = specs.split(';')
      for i, word in enumerate(encoders_list_specs):
        nodes_quality.append[word[0]]
        encoders_list_specs[i] = word[1:]
      
      it = iter(encoders_list_specs)
      output = self._initialize_single_encoder_from_spec(next(it), quals=nodes_quality)
      while True:
        try:
          nxt = next(it)
          self._initialize_single_encoder_from_spec(nxt, quals=nodes_quality, inputs=output)
        except StopIteration:
          break
    else:
      raise ValueError("")

  def _initialize_single_encoder_from_spec(self, spec, quals, inputs=None):
    """
    """
    if inputs is None: inputs = self.Y
    
    
  
  
  def train(self):
    """
    """
    raise NotImplementedError("Method `train` is not defined in user class")