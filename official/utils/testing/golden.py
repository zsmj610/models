# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""TensorFlow testing subclass to automate numerical testing.

  Golden tests determine when behavior deviates from some "gold standard", and
are useful for determining when layer definitions have changed without
performing full regression testing, which is generally prohibitive. The tests
performed by this class are:

1) Compare a generated graph against a reference graph. Differences are not
   necessarily fatal.
2) Attempt to load known weights for the graph. If this step succeeds but
   changes are present in the graph, a warning is issued but does not raise
   an exception.
3) Perform a calculation and compare the result to a reference value.

This class also provides a method to generate reference data.

Note:
  The user is responsible for fixing the random seed during graph definition. A
  convenience method name_to_seed() is provided to make this process easier.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import hashlib
import json
import os
import shutil
import warnings


import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


class BaseTest(tf.test.TestCase):
  """TestCase subclass for performing golden tests.
  """

  @property
  def data_root(self):
    return

    # # In subclass use:
    # return os.path.join(os.path.split(
    #     os.path.abspath(__file__))[0], "test_data")

  @property
  def ckpt_prefix(self):
    return "model.ckpt"

  @staticmethod
  def name_to_seed(name):
    seed = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(seed, 16) % (2**32 - 1)

  @staticmethod
  def common_tensor_properties(input_array):
    """Convenience function for matrix testing.

    Args:
      input_array: Tensor (numpy array), from which key values are extracted.

    Returns:
      A list of key values.
    """
    output = list(input_array.shape)
    flat_array = input_array.flatten()
    output.extend([float(i) for i in
                   [flat_array[0], flat_array[-1], np.sum(flat_array)]])
    return output

  def default_correctness_function(self, *args):
    output = []
    for arg in args:
      output.extend(self.common_tensor_properties(arg))
    return output

  def _construct_test_case(self, name, g, ops_to_eval, correctness_function):
    data_dir = os.path.join(self.data_root, name)

    # Make sure there is a clean space for results.
    if os.path.exists(data_dir):
      shutil.rmtree(data_dir)
    os.mkdir(data_dir)

    # Serialize graph for comparison.
    graph_bytes = g.as_graph_def().SerializeToString()
    expected_file = os.path.join(data_dir, "expected_graph")
    with open(expected_file, "wb") as f:
      f.write(graph_bytes)

    with g.as_default():
      init = tf.global_variables_initializer()
      saver = tf.train.Saver()

    with self.test_session(graph=g) as sess:
      sess.run(init)
      saver.save(sess=sess, save_path=os.path.join(data_dir, self.ckpt_prefix))

      # These files are not needed for this test.
      os.remove(os.path.join(data_dir, "checkpoint"))
      os.remove(os.path.join(data_dir, self.ckpt_prefix + ".meta"))

      ops = [op.eval() for op in ops_to_eval]
      if correctness_function is not None:
        results = correctness_function(*ops)
        with open(os.path.join(data_dir, "results.json"), "wt") as f:
          json.dump(results, f)

  def _evaluate_test_case(self, name, g, ops_to_eval, correctness_function):
    data_dir = os.path.join(self.data_root, name)

    # Serialize graph for comparison.
    graph_bytes = g.as_graph_def().SerializeToString()
    expected_file = os.path.join(data_dir, "expected_graph")
    with open(expected_file, "rb") as f:
      expected_graph_bytes = f.read()
      # The serialization is non-deterministic byte-for-byte. Instead there is
      # a utility which evaluates the semantics of the two graphs to test for
      # equality. This has the added benefit of providing some information on
      # what changed.
      #   Note: The summary only show the first difference detected. It is not
      #         an exhaustive summary of differences.
    differences = pywrap_tensorflow.EqualGraphDefWrapper(
        graph_bytes, expected_graph_bytes).decode("utf-8")

    with g.as_default():
      init = tf.global_variables_initializer()
      saver = tf.train.Saver()

    with self.test_session(graph=g) as sess:
      sess.run(init)
      try:
        saver.restore(sess=sess, save_path=os.path.join(
            data_dir, self.ckpt_prefix))
        if differences:
          print()
          warnings.warn(
              "The provided graph is different than expected:\n  {}\n"
              "However the weights were still able to be loaded.\n".format(
                  differences)
          )
      except:  # pylint: disable=bare-except
        raise self.failureException("Weight load failed. Graph comparison:\n  "
                                    "{}".format(differences))

      ops = [op.eval() for op in ops_to_eval]
      if correctness_function is not None:
        results = correctness_function(*ops)
        with open(os.path.join(data_dir, "results.json"), "rt") as f:
          expected_results = json.load(f)
        self.assertAllClose(results, expected_results)

  def _manage_ops(self, name, g, ops_to_eval=None, test=True,
                  correctness_function=None):
    """Utility function to handle repeated work of graph checking.

    Args:
      name: String defining the run. This will be used to define folder names
        and will be used for random seed construction.
      g: The graph in which the test is conducted.
      ops_to_eval: Ops which the user wishes to be evaluated under a controlled
        session.
      test: Boolean. If True this function will test graph correctness, load
        weights, and compute numerical values. If False the necessary test data
        will be generated and saved.
      correctness_function: This function accepts the evaluated results of
        ops_to_eval, and returns a list of values. This list must be JSON
        serializable; in particular it is up to the user to convert numpy
        dtypes into builtin dtypes.
    """

    ops_to_eval = [] if ops_to_eval is None else ops_to_eval

    if test:
      self._evaluate_test_case(
          name=name, g=g, ops_to_eval=ops_to_eval,
          correctness_function=correctness_function
      )
    else:
      self._construct_test_case(
          name=name, g=g, ops_to_eval=ops_to_eval,
          correctness_function=correctness_function
      )
