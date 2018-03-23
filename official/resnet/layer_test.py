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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os


import tensorflow as tf
from official.resnet import resnet_model   # pylint: disable=g-bad-import-order
from official.utils.testing import golden  # pylint: disable=g-bad-import-order


DATA_FORMAT = "channels_last"  # CPU instructions often preclude channels_last


class BaseTest(golden.BaseTest):
  """Tests for core ResNet layers.
  """

  @property
  def data_root(self):
    """Use the subclass directory rather than the parent directory.

    Returns:
      The path prefix for reference data.
    """
    return os.path.join(os.path.split(
        os.path.abspath(__file__))[0], "test_data")

  def _batch_norm_ops(self, test=False):
    name = "batch_norm"

    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(self.name_to_seed(name))
      input_tensor = tf.get_variable(
          "input_tensor", dtype=tf.float32,
          initializer=tf.random_uniform((32, 16, 16, 3), maxval=1)
      )
      layer = resnet_model.batch_norm(
          inputs=input_tensor, data_format=DATA_FORMAT, training=True)

    self._manage_ops(
        name=name, g=g, ops_to_eval=[input_tensor, layer], test=test,
        correctness_function=self.default_correctness_function
    )

  def make_projection(self, filters_out, strides, data_format):
    """1D convolution with stride projector.

    Args:
      filters_out: Number of filters in the projection.
      strides: Stride length for convolution.
      data_format: channels_first or channels_last

    Returns:
      A 1 wide CNN projector function.
    """
    def projection_shortcut(inputs):
      return resnet_model.conv2d_fixed_padding(
          inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
          data_format=data_format)
    return projection_shortcut

  def _resnet_block_ops(self, test, batch_size, bottleneck, projection,
                        version, width, channels):
    """Test whether resnet block construction has changed.

    Args:
      test: Whether or not to run as a test case.
      batch_size: Number of points in the fake image. This is needed due to
        batch normalization.
      bottleneck: Whether or not to use bottleneck layers.
      projection: Whether or not to project the input.
      version: Which version of ResNet to test.
      width: The width of the fake image.
      channels: The number of channels in the fake image.
    """

    name = "batch_size_{}__{}{}__version_{}__width_{}__channels_{}".format(
        batch_size, "bottleneck" if bottleneck else "building",
        "__projection" if projection else "", version, width, channels
    )

    if version == 1:
      block_fn = resnet_model._building_block_v1
      if bottleneck:
        block_fn = resnet_model._bottleneck_block_v1
    else:
      block_fn = resnet_model._building_block_v2
      if bottleneck:
        block_fn = resnet_model._bottleneck_block_v2

    g = tf.Graph()
    with g.as_default():
      tf.set_random_seed(self.name_to_seed(name))
      strides = 1
      channels_out = channels
      projection_shortcut = None
      if projection:
        strides = 2
        channels_out *= strides
        projection_shortcut = self.make_projection(
            filters_out=channels_out, strides=strides, data_format=DATA_FORMAT)

      filters = channels_out
      if bottleneck:
        filters = channels_out // 4

      input_tensor = tf.get_variable(
          "input_tensor", dtype=tf.float32,
          initializer=tf.random_uniform((batch_size, width, width, channels),
                                        maxval=1)
      )

      layer = block_fn(inputs=input_tensor, filters=filters, training=True,
                       projection_shortcut=projection_shortcut, strides=strides,
                       data_format=DATA_FORMAT)

    self._manage_ops(
        name=name, g=g, ops_to_eval=[input_tensor, layer], test=test,
        correctness_function=self.default_correctness_function
    )

  def test_batch_norm(self):
    self._batch_norm_ops(test=True)

  def test_bottleneck_v1_width_8_channels_16_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    self._resnet_block_ops(test=True, batch_size=32, bottleneck=True,
                           projection=True, version=1, width=8, channels=16)

  def test_bottleneck_v2_width_8_channels_16_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    self._resnet_block_ops(test=True, batch_size=32, bottleneck=True,
                           projection=True, version=2, width=8, channels=16)

  def test_bottleneck_v1_width_8_channels_16_batch_size_32(self):
    """Test of a single ResNet block."""
    self._resnet_block_ops(test=True, batch_size=32, bottleneck=True,
                           projection=False, version=1, width=8, channels=16)

  def test_bottleneck_v2_width_8_channels_16_batch_size_32(self):
    """Test of a single ResNet block."""
    self._resnet_block_ops(test=True, batch_size=32, bottleneck=True,
                           projection=False, version=2, width=8, channels=16)

  def test_building_v1_width_8_channels_16_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    self._resnet_block_ops(test=True, batch_size=32, bottleneck=False,
                           projection=True, version=1, width=8, channels=16)

  def test_building_v2_width_8_channels_16_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    self._resnet_block_ops(test=True, batch_size=32, bottleneck=False,
                           projection=True, version=2, width=8, channels=16)

  def test_building_v1_width_8_channels_16_batch_size_32(self):
    """Test of a single ResNet block."""
    self._resnet_block_ops(test=True, batch_size=32, bottleneck=False,
                           projection=False, version=1, width=8, channels=16)

  def test_building_v2_width_8_channels_16_batch_size_32(self):
    """Test of a single ResNet block."""
    self._resnet_block_ops(test=True, batch_size=32, bottleneck=False,
                           projection=False, version=2, width=8, channels=16)

  def regenerate(self):
    self._batch_norm_ops(test=False)
    self._resnet_block_ops(test=False, batch_size=32, bottleneck=True,
                           projection=True, version=1, width=8, channels=16)

    self._resnet_block_ops(test=False, batch_size=32, bottleneck=True,
                           projection=True, version=2, width=8, channels=16)

    self._resnet_block_ops(test=False, batch_size=32, bottleneck=True,
                           projection=False, version=1, width=8, channels=16)

    self._resnet_block_ops(test=False, batch_size=32, bottleneck=True,
                           projection=False, version=2, width=8, channels=16)

    self._resnet_block_ops(test=False, batch_size=32, bottleneck=False,
                           projection=True, version=1, width=8, channels=16)

    self._resnet_block_ops(test=False, batch_size=32, bottleneck=False,
                           projection=True, version=2, width=8, channels=16)

    self._resnet_block_ops(test=False, batch_size=32, bottleneck=False,
                           projection=False, version=1, width=8, channels=16)

    self._resnet_block_ops(test=False, batch_size=32, bottleneck=False,
                           projection=False, version=2, width=8, channels=16)


if __name__ == "__main__":
  tf.test.main()
