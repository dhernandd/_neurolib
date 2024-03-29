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
import numpy as np
import tensorflow as tf

DTYPE = tf.float32

# pylint: disable=bad-indentation, no-member, protected-access

def variable_in_cpu(name, shape, initializer, collections=None):
    """
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name,
                              shape,
                              dtype=DTYPE,
                              initializer=initializer,
                              collections=collections)
    return var


make_var_name = lambda scope, name : scope + '/' + name + ':0' 


def make_data_iterator(data, batch_size=1, shuffle=True):
  """
  Iterate over data (simple)
  
  Args:
    TODO:
  """
  nsamps = len(data[0])
  l_inds = np.arange(nsamps)
  if shuffle: 
    np.random.shuffle(l_inds)

  for i in range(0, nsamps, batch_size):
    yield [ d[l_inds[i:i+batch_size]] for d in data ]
        
        
def check_name(f):
  """
  Check that the name of a node has not been taken
  """
  def f_checked(obj, *args, **kwargs):
    if 'name' in kwargs:
      if kwargs['name'] in obj.nodes:
        raise AttributeError("The name", kwargs["name"], "already corresponds "
                             "to a node in this graph")
    return f(obj, *args, **kwargs)
  
  return f_checked

def basic_concatenation(_input):
  """
  """
  try:
    _input = tf.concat(_input, axis=-1)
  except ValueError:
    itensors = list(zip(*sorted(_input.items())))[1] # mae sure inputs are ordered
    _input = tf.concat(itensors, axis=-1)
  
  return _input