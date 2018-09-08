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
# from tensorflow.contrib import distributions as _distributions
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow_probability import distributions as _distributions

import inspect

_globals = globals()
for dist_name in sorted(dir(_distributions)):
  _candidate = getattr(_distributions, dist_name)
  if (inspect.isclass(_candidate) and
          _candidate != _distributions.Distribution and
          issubclass(_candidate, _distributions.Distribution)):
    
    # All distributions are imported here
    _globals[dist_name] = getattr(_distributions, dist_name)

    del _candidate  

__all__ = ["MultivariateNormalTriL"]

# print(sorted(globals().keys()))