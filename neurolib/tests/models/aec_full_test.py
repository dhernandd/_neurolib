
import unittest
import tensorflow as tf

from neurolib.models.aec import BasicAE

class BasicAEFullTest(tf.test.TestCase):
  """
  """
  default_specs = {'b_size' : 1}
  default_ydim = 10
  default_xdim = 2
  def test_init(self):
    """
    """
    tf.reset_default_graph()
    ae = BasicAE(self.default_ydim, self.default_xdim,
                 self.default_specs)
    
  def test_sample(self):
    """
    """
    tf.reset_default_graph()  # Worry about working with multiple graphs
    ae1 = BasicAE(10, 2, self.default_specs)
    ysamps1 = ae1.sample()
    print(ysamps1[0:2])

#   def test_sample2(self):
#     tf.reset_default_graph()
#     ae2 = BasicAE(10, 2, self.default_specs)
#     ysamps2 = ae2.sample()
#     print(ysamps2[0:2])
    
  def test_train(self):
    """
    """
    tf.reset_default_graph()
    ae1 = BasicAE(10, 2, self.default_specs)
    ysamps = ae1.sample()
    
    tf.reset_default_graph()
    ae2 = BasicAE(10, 2, self.default_specs)
    ae2.train(ysamps, 5)
    

if __name__ == '__main__':
  unittest.main()
#   tf.test.main()