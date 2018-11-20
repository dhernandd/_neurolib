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

import numpy as np
import tensorflow as tf

from neurolib.encoder.basic import InnerNode
from neurolib.encoder import act_fn_dict, layers_dict, cell_dict
from neurolib.encoder.input import NormalInputNode
from neurolib.encoder.seq_cells import CustomCell

# pylint: disable=bad-indentation, no-member, protected-access
    
class EvolutionSequence(InnerNode):
  """
  An EvolutionSequence represents a sequence of mappings, each with the
  distinguishining feature that it takes the output of their predecessor as
  input. This makes them appropriate in particular to represent the time
  evolution of a code.
  
  RNNs are children of EvolutionSequence.
  """
  num_expected_outputs = 1
  _requires_builder = True
  def __init__(self,
               label,
               num_features,
               init_states=None,
               num_islots=2,
               max_steps=30,
               batch_size=1,
               name=None,
               builder=None,
               mode='forward'):
    """
    Initialize an EvolutionSequence
    """
    self.name = 'EvSeq_' + str(label) if name is None else name    
    super(EvolutionSequence, self).__init__(label)

    self.num_features = num_features
    self.max_steps = max_steps
    self.batch_size = batch_size
    self.main_oshape = [batch_size, max_steps, num_features]
    self._oslot_to_shape[0] = self.main_oshape
    
    if init_states is None:
      raise ValueError("`init_states` must be provided") 
    
    self.free_oslots = list(range(self.num_expected_outputs))

    self.builder = builder
    self.mode = mode
    self.num_expected_inputs = num_islots
    
  @abstractmethod
  def _build(self):
    """
    """
    raise NotImplementedError("Please implement me.")


class BasicRNNEvolutionSequence(EvolutionSequence):
  """
  BasicRNNEvolutionSequence is the simplest possible EvolutionSequence. It is a
  mapping whose inputs are an external input tensor and the previous state of
  the sequence, and whose output is the (i+1)th state. 
  """
  def __init__(self,
               label, 
               num_features,
               init_states,
               num_islots=2,
               max_steps=30,
               batch_size=1,
               name=None,
               builder=None,
               cell='basic',
               mode='forward',
               **dirs):
    """
    Initialize the BasicRNNEvolutionSequence
    """
    super(BasicRNNEvolutionSequence, self).__init__(label,
                                                    num_features,
                                                    init_states=init_states,
                                                    num_islots=num_islots,
                                                    max_steps=max_steps,
                                                    batch_size=batch_size,
                                                    name=name,
                                                    builder=builder,
                                                    mode=mode)
    if len(init_states) != 1:
      raise ValueError("`len(init_states) != 1`")
    if num_features != init_states[0].num_features:
      raise ValueError("num_features != init_states.num_features, {} != {}",
                       num_features, init_states.num_features)

    if isinstance(init_states[0], str):
      self.init_state = builder.nodes[init_states]
    else:
      self.init_state = init_states[0]
    
    self.cell = cell if not isinstance(cell, str) else cell_dict['cell']
    builder.addDirectedLink(self.init_state, self, islot=0)
    self._update_default_directives(**dirs)
    
  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {}
    self.directives.update(dirs)
    
#     if isinstance(self.directives['cell'], str): 
#       self.directives['cell'] = cell_dict[self.directives['cell']]
    
  def _build(self):
    """
    Build the Evolution Sequence
    """
    sorted_inputs = sorted(self._islot_to_itensor.items())
    
    init_state = sorted_inputs[0][1]
    inputs_series = tuple(zip(*sorted_inputs[1:]))[1]
#     if len(inputs_series) == 1:
#       inputs_series = inputs_series[0]
#     else:
#       inputs_series = tf.concat(inputs_series, axis=-1)
    
    rnn_cell = self.cell
#     rnn_cell = self.directives['cell']
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      if issubclass(rnn_cell, CustomCell): # TODO: I am diverging here from the tf API, is this really necessary? 
        cell = rnn_cell(self.num_features,
                        builder=self.builder)  #pylint: disable=not-callable
      else:
        rnn_cell(self.num_features)
      
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
    
    self._oslot_to_otensor[0] = tf.identity(states_series, name=self.name)
    
    self._is_built = True


class LSTMEvolutionSequence(EvolutionSequence):
  """
  """
  def __init__(self,
               label, 
               num_features,
               init_states,
               num_islots=3,
               max_steps=30,
               batch_size=1,
               name=None,
               builder=None,
               mode='forward',
               **dirs):
    """
    Initialize the LSTMEvolutionSequence
    """
    super(LSTMEvolutionSequence, self).__init__(label,
                                                num_features,
                                                init_states=init_states,
                                                num_islots=num_islots,
                                                max_steps=max_steps,
                                                batch_size=batch_size,
                                                name=name,
                                                builder=builder,
                                                mode=mode)
    
    self.init_state, self.init_hidden_state = init_states[0], init_states[1]
    builder.addDirectedLink(self.init_state, self, islot=0)
    builder.addDirectedLink(self.init_hidden_state, self, islot=1)
    
    self._update_default_directives(**dirs)

  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {'cell' : 'lstm'}
    self.directives.update(dirs)
    
    self.directives['cell'] = cell_dict[self.directives['cell']]
    
  def _build(self):
    """
    Build the Evolution Sequence
    """
    sorted_inputs = sorted(self._islot_to_itensor.items())
    
    init_state = tf.nn.rnn_cell.LSTMStateTuple(sorted_inputs[0][1], sorted_inputs[1][1])
    
    inputs_series = tuple(zip(*sorted_inputs[2:]))[1]
    if len(inputs_series) == 1:
      inputs_series = inputs_series[0]
    else:
      inputs_series = tf.concat(inputs_series, axis=-1)
    
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_features)  #pylint: disable=not-callable
      
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
    
    self._oslot_to_otensor[0] = tf.identity(states_series, name=self.name)
    
    self._is_built = True


class LinearNoisyDynamicsEvSeq(EvolutionSequence):
  """
  """
  def __init__(self,
               label, 
               num_features,
               init_states,
               num_islots=1,
               max_steps=30,
               batch_size=1,
               name=None,
               builder=None,
               mode='forward',
               **dirs):
    """
    """
    self.init_state = init_states[0]
    super(LinearNoisyDynamicsEvSeq, self).__init__(label,
                                                   num_features,
                                                   init_states=init_states,
                                                   num_islots=num_islots,
                                                   max_steps=max_steps,
                                                   batch_size=batch_size,
                                                   name=name,
                                                   builder=builder,
                                                   mode=mode)

    builder.addDirectedLink(self.init_state, self, islot=0)
    self._update_default_directives(**dirs)
    
  def _update_default_directives(self, **dirs):
    """
    Update the default directives
    """
    self.directives = {}
    self.directives.update(dirs)
        
  def _build(self):
    """
    Build the Evolution Sequence
    """
    sorted_inputs = sorted(self._islot_to_itensor.items())
    
    print("sorted_inputs", sorted_inputs)
    init_state = sorted_inputs[0][1]
    inputs_series = tuple(zip(*sorted_inputs[1:]))[1]
    if len(inputs_series) == 1:
      inputs_series = inputs_series[0]
    else:
      inputs_series = tf.concat(inputs_series, axis=-1)
    print("self.num_features", self.num_features)
    
    rnn_cell = self.directives['cell']
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      cell = rnn_cell(self.num_features)  #pylint: disable=not-callable
      
      print("inputs_series", inputs_series)
      print("init_state", init_state)
      states_series, _ = tf.nn.dynamic_rnn(cell,
                                           inputs_series,
                                           initial_state=init_state)
      print("states_series", states_series)
    
    self._oslot_to_otensor[0] = tf.identity(states_series, name=self.name)
    
    self._is_built = True
    
    
class CustomEvolutionSequence():
  """
  """
  pass