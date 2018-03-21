# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Provides utilities for exporting models in various formats."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def build_tensor_serving_input_receiver_fn(shape, dtype=tf.float32,
                                           batch_size=1):
  """Returns a input_receiver_fn that can be used during serving.

  This expects examples to come through as float tensors, and simply
  wraps them as TensorServingInputReceivers.

  Args:
    shape: list representing target size of a single example.
    dtype: the expected datatype for the input example
    batch_size: number of input tensors that will be passed for prediction

  Returns:
    A function that itself returns a TensorServingInputReceiver.
  """
  def serving_input_receiver_fn():
    # Prep a placeholder where the input example will be fed in
    processed_image = tf.placeholder(
        dtype=dtype, shape=[batch_size] + shape, name='input_tensor')

    return tf.estimator.export.TensorServingInputReceiver(
        features=processed_image, receiver_tensors=processed_image)

  return serving_input_receiver_fn


def export_model(estimator, export_dir, shape, dtype=tf.float32, batch_size=1):
  """Exports a model to the specified directory.

  Args:
    estimator: a trained instance of tf.estimator.Estimator for exporting.
    export_dir: String path to the target directory for export.
    shape: list representing target size of a single example.
    dtype: the expected datatype for the input example
    batch_size: number of input tensors that will be passed for prediction

  Returns:
    The estimator in question.

  Raises:
    ValueError: if export_dir is None.
  """
  if export_dir is None:
    raise ValueError('An export_dir must be passed in to export a model.')

  input_receiver_fn = build_tensor_serving_input_receiver_fn(
      shape, dtype, batch_size)
  estimator.export_savedmodel(export_dir, input_receiver_fn)

  return estimator
