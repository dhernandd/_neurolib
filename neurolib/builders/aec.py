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
from neurolib.builders.static_builder import StaticModelBuilder

class AECBuilder(StaticModelBuilder):
  """
  """
  def __init__(self):
    """
    """
    super(AECBuilder, self).__init__()
    
  def build(self, specs):
    """
    TODO: Deal with user-provided specs
    """
    if 'graph_directive' not in specs:
      pass
    else:
      pass
    
    return