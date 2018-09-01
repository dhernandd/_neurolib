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

from neurolib.utils.graphs import get_session

# WE REFUSE TO SUPPORT PYTHON 2! :D
class Model(abc.ABC):
  """
  TODO: Put the abc functionality to use.
  """
  def __init__(self):
    """
    TODO: Should I start a session here? This presents some troubles, at least
    with this implementation of get_session which I am beginning to suspect it
    is not going to cut it for our purposes. The sessions needs to be
    micromanaged...
    
    TODO: I also want to manually manage the graphs for when people want to run
    two models and compare for example.
    """
#     sess = get_session()
    pass

  
  def _build(self):
    """
    TODO: Fill the exception
    """
    raise NotImplementedError("")
    
  @abc.abstractmethod
  def train(self):
    """
    TODO: Fill the exception
    """
    raise NotImplementedError("")
  
  def sample(self):
    """
    TODO: Think about what is meant by sample. Only Generative Models should
    have sampling capabilities so this function probably shouldnt be here since
    I am defining a `Model` as the most general composition of Encoders. sample
    in fact seems to be a method of the abstract Encoder class, not of the
    Model, huh?
    
    TODO: Fill the exception
    """
    raise NotImplementedError("")
  

#### DUMP DUMP DUMP (For now) ####
class Unsupervised():
  """
  TODO: This is for the moment an experiment on the hierarchy below Model.
  Nothing serious here right now.
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