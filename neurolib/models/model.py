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

  def _initialize_from_specs(self):
    """
    """
    raise NotImplementedError("")

  def update(self):
    """
    TODO: Fill
    """
    raise NotImplementedError("")
  
  def train(self):
    """
    """
    raise NotImplementedError("Method `train` is not defined in user class")