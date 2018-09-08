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
import unittest
import tensorflow as tf

from neurolib.models.lindet import BasicAE, BayesianAE

class BasicAEFullTest(tf.test.TestCase):
  """
  TODO: Write these in terms of self.Assert...
  """
  default_specs = {'b_size' : 1}
  default_ydim = 10
  default_xdim = 2
  
  @unittest.skip("Skipping")
  def test_init(self):
    """
    """
    tf.reset_default_graph()
    ae = BasicAE(self.default_ydim, self.default_xdim, self.default_specs)
    
  @unittest.skip("Skipping")
  def test_sample(self):
    """
    """
    tf.reset_default_graph()  # Worry about working with multiple graphs
    ae1 = BasicAE(10, 2, self.default_specs)
    ysamps1 = ae1.sample()
    print(ysamps1[0:2])
# 
# #   def test_sample2(self):
# #     tf.reset_default_graph()
# #     ae2 = BasicAE(10, 2, self.default_specs)
# #     ysamps2 = ae2.sample()
# #     print(ysamps2[0:2])
# #     

  @unittest.skip("Skipping")
  def test_train(self):
    """
    """
    tf.reset_default_graph()
    ae1 = BasicAE(10, 2, self.default_specs)
    ysamps = ae1.sample()
      
    tf.reset_default_graph()
    ae2 = BasicAE(10, 2, self.default_specs)
    ae2.train(ysamps, 5)
      
    
class BayesianAEFullTest(tf.test.TestCase):
  """
  """
  default_specs = {'b_size' : 1}
  default_ydim = 10
  default_xdim = 2

  def test_init(self):
    """
    """
    tf.reset_default_graph()
    ae = BayesianAE(self.default_ydim, self.default_xdim, self.default_specs)

  def test_train(self):
    """
    """
    tf.reset_default_graph()
    ae1 = BayesianAE(10, 2, self.default_specs)
    ysamps = ae1.sample()
    print('Shape samps:', ysamps.shape)
      
    tf.reset_default_graph()
    ae2 = BayesianAE(10, 2, self.default_specs)
    chol = ae2.enc_Yprime.cholesky_tril
    chol_inv = tf.matrix_triangular_solve(chol, tf.reshape(tf.eye(10), [1, 10, 10]))
    input_name = ae2.enc_H1.get_input().name
    print(input_name)
    
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      print(sess.run(chol, feed_dict={input_name : ysamps[0:1]}))
      print(sess.run(chol_inv, feed_dict={input_name : ysamps[0:1]}))
      ae2.train(ysamps, 100)

if __name__ == '__main__':
  unittest.main()
#   tf.test.main()