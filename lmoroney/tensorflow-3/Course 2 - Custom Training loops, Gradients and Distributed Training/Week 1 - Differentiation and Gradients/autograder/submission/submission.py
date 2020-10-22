# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% colab={} colab_type="code" id="i_MD3wdwHFn4"
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% colab={} colab_type="code" id="jqev488WJ9-R"
import tensorflow as tf
import numpy as np

# %% [markdown] colab_type="text" id="sIIXMZboUw-P"
# # Exercise on basic Tensor operations

# %% colab={} colab_type="code" id="GkBZyS8hKNXX"
# Create a 1D uint8 NumPy array comprising of first 9 natural numbers
x = np.arange(1, 10)
x


# %% colab={} colab_type="code" id="MYdVyiSoLPgO"
# Convert NumPy array to Tensor using `tf.constant`
def tf_constant(array):
    
    ### START CODE HERE ###
    tf_constant_array = tf.constant(array)
    ### END CODE HERE ###
    return tf_constant_array


# %%
x = tf_constant(x)
x

# %%
unit_tests.test_tf_constant(tf_constant)


# %% colab={} colab_type="code" id="W6BTwNJCLjV8"
# Square the input tensor x
def tf_square(array):
    
    ### START CODE HERE ###
    tf_squared_array = tf.square(array)
    ### END CODE HERE ###
    return tf_squared_array


# %%
x = tf_square(x)
x


# %% colab={} colab_type="code" id="7nzBSX8-L0Xt"
# Reshape tensor x into a 3 x 3 matrix
def tf_reshape(array, shape):
    
    ### START CODE HERE ###
    tf_reshaped_array = tf.reshape(array, shape)
    ### END CODE HERE ###
    return tf_reshaped_array


# %%
x = tf_reshape(x, (3, 3))
x


# %% colab={} colab_type="code" id="VoT-jiAIL8x5"
# Cast tensor x into float32 
def tf_cast(array, dtype):
    
    ### START CODE HERE ###
    tf_cast_array = tf.cast(array, dtype)
    ### END CODE HERE ###
    return tf_cast_array


# %%
x = tf_cast(x, tf.float32)
x

# %% colab={} colab_type="code" id="VjnvT4LLL_5K"
y = tf.constant(2, dtype=tf.float32)
y


# %% colab={} colab_type="code" id="ivepGtD5MKP5"
# Multiply tensor x and y
def tf_multiply(num1, num2):
    
    ### START CODE HERE ###
    product = tf.multiply(num1, num2)
    ### END CODE HERE ###
    return product



# %%
result = tf_multiply(x, y)
result.numpy()


# Expected output:
# [[  2.,   8.,  18.],
#  [ 32.,  50.,  72.],
#  [ 98., 128., 162.]]

# %% colab={} colab_type="code" id="8wzZ5NcMMPzD"
y = tf.constant([1, 2, 3], dtype=tf.float32)
y


# %% colab={} colab_type="code" id="BVlntdYnMboh"
# Add tensor x and y
def tf_add(num1, num2):
    
    ### START CODE HERE ###
    sum = num1 + num2
    ### END CODE HERE ###
    return sum


# %%
result = tf_add(x, y)
result

# Expected output:
# [[ 3.,  6., 11.],
#  [18., 27., 38.],
#  [51., 66., 83.]]

# %% [markdown] colab_type="text" id="9EN0W15EWNjD"
# # Exercise on Gradient Tape

# %% colab={} colab_type="code" id="p3K94BWZM6nW"
def tf_gradient_tape(x):
    
    with tf.GradientTape() as t:
        
    ### START CODE HERE ###
        # Record the actions performed on tensor x with `watch`
        t.watch(x)		    

        # Define a polynomial of form 3x^3 - 2x^2 + x
        y =  3 * (x ** 3) - 2 * (x ** 2)  + x    

        # Obtain the sum of variable y
        z = tf.reduce_sum(y) 
  
    # Derivative of z wrt the original input tensor x
    dz_dx = t.gradient(z, x)
    ### END CODE HERE
    
    return dz_dx


# %% colab={} colab_type="code" id="MKy-PXrPPecb"
x = tf.constant(2.0)

tf_gradient_tape(x)

# %%
# Convert dz_dx into NumPy 
result = dz_dx.numpy()
result

# Expected output:
# 29.0
