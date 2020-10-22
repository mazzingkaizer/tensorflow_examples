# -*- coding: utf-8 -*-
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

# %% [markdown] colab_type="text" id="n4EKOpw9mObL"
# ## Setup
#
# Import TensorFlow 2.0:

# %% colab={} colab_type="code" id="V9oECvVSI1Kj"
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

# %% colab={} colab_type="code" id="mT7meGqrZTz9"
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt

# %% [markdown] colab_type="text" id="NfpIQUv28Ht4"
# ## Exercise on in-graph training loop
#
# This exercise teaches how to train a Keras model on the `horses_or_humans` dataset with the entire training process—loading batches, calculating gradients, updating parameters, calculating validation accuracy, and repeating until convergence— performed in-graph.

# %% [markdown] colab_type="text" id="Em5dzSUOtLRP"
# ### Prepare the dataset

# %%
splits, info = tfds.load('horses_or_humans', as_supervised=True, with_info=True, split=['train[:80%]', 'train[80%:]', 'test'], data_dir='../data')

(train_examples, validation_examples, test_examples) = splits

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

# %% colab={} colab_type="code" id="cJdruxxGhBi5"
IMAGE_SIZE = 224
BATCH_SIZE = 32


# %%
def set_image_size(size):
    image_size = size
    return image_size


# %% colab={} colab_type="code" id="qpQi4Jo9cFq0"
# Create a autograph pre-processing function to resize and normalize an image
### START CODE HERE ###
@tf.function
def map_fn(img, label):
    image_size = set_image_size(224) # @REPLACE image_size = set_image_size(224)
    # resize the image
    img = tf.image.resize(img, (image_size, image_size))
    # normalize the image
    img /= 255.0
### END CODE HERE
    return img, label


# %% colab={} colab_type="code" id="sv5bEYhaeUUO"
# Prepare train dataset by using preprocessing with map_fn, shuffling and batching
def prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn, batch_size):
    ### START CODE HERE ###
    train_ds = train_examples.map(map_fn).shuffle(buffer_size=num_examples).batch(batch_size)
    ### END CODE HERE ###
    valid_ds = validation_examples.map(map_fn).batch(batch_size)
    test_ds = test_examples.map(map_fn).batch(batch_size)
    
    return train_ds, valid_ds, test_ds


# %%
train_ds, valid_ds, test_ds = prepare_dataset(train_examples, validation_examples, test_examples, num_examples, map_fn, BATCH_SIZE)

# %% [markdown] colab_type="text" id="znmy4l8ntMvW"
# ### Define the model

# %% colab={} colab_type="code" id="ltxyJVWTqNAO"
MODULE_HANDLE = '../data/resnet_50_feature_vector'
model = tf.keras.Sequential([
    hub.KerasLayer(MODULE_HANDLE, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.summary()


# %% [markdown] colab_type="text" id="Ikb79EzkjpPk"
# ## Define optimizer, loss and metrics

# %%
def set_adam_optimizer():
    ### START CODE HERE ###
    # Define the adam optimizer
    optimizer = tf.keras.optimizers.Adam()
    ### END CODE HERE ###
    return optimizer


# %%
def set_sparse_cat_crossentropy_loss():
    ### START CODE HERE ###
    # Define object oriented metric of Sparse categorical crossentropy for train and val loss
    train_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    val_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    ### END CODE HERE ###
    return train_loss, val_loss


# %%
def set_sparse_cat_crossentropy_accuracy():
    ### START CODE HERE ###
    # Define object oriented metric of Sparse categorical accuracy for train and val accuracy
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    ### END CODE HERE ###
    return train_accuracy, val_accuracy


# %% colab={} colab_type="code" id="j92oDYGCjnBh"
optimizer = set_adam_optimizer()
train_loss, val_loss = set_sparse_cat_crossentropy_loss()
train_accuracy, val_accuracy = set_sparse_cat_crossentropy_accuracy()

# %% [markdown] colab_type="text" id="oeYV6mKnJGMr"
# ### Define the training loop

# %% colab={} colab_type="code" id="rnGS06sDuaIy"
device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'

# %% colab={} colab_type="code" id="5JRbx72axEoM"
EPOCHS = 2


# %% colab={} colab_type="code" id="3xtg_MMhJETd"
# Custom training step
def train_one_step(model, optimizer, x, y, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
    ### START CODE HERE ###
        # Run the model on input x to get predictions
        predictions = model(x)
        # Compute the training loss using `train_loss` 
        loss = train_loss(y, predictions)

    # Using the tape and loss, compute the gradients on model variables
    grads = tape.gradient(loss, model.trainable_variables)
    
    # Zip the gradients and model variables, and then apply the result on the optimizer
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Call the train accuracy object on ground truth and predictions
    train_accuracy(y, predictions)
    ### END CODE HERE
    return loss


# %%
# Decorate this function with tf.function to enable autograph on the training loop
@tf.function
def train(model, optimizer, epochs, device, train_ds, train_loss, train_accuracy, valid_ds, val_loss, val_accuracy):
    step = 0
    loss = 0.0
    for epoch in range(epochs):
        for x, y in train_ds:
            step += 1
            with tf.device(device_name=device):
                ### START CODE HERE ###
                # Run one training step by passing appropriate model parameters
                # required by the function and finally get the loss to report the results
                loss = train_one_step(model, optimizer, x, y, train_loss, train_accuracy)
            # Rely on reliable debugging functions like tf.print to report your results.
            # Print the training step number, loss and accuracy
            tf.print('Step', step, # @REPLACE             None('Step', None,
                   ': train loss', loss, # @REPLACE                    ': train loss', None,
                   '; train accuracy', train_accuracy.result()) # @REPLACE                    '; train accuracy', None)
            ### END CODE HERE ###

        with tf.device(device_name=device):
            for x, y in valid_ds:
                # Call the model on the batches of inputs x and get the predictions
                y_pred = model(x)
                loss = val_loss(y, y_pred)
                val_accuracy(y, y_pred)
        
        ### START CODE HERE ###
        # Print the validation loss and accuracy
        tf.print('val loss', loss, '; val accuracy', val_accuracy.result()) # @REPLACE        None('val loss', None, '; val accuracy', None)
        ### END CODE HERE


# %% colab={} colab_type="code" graded=true id="6iDWgg977wb9" name="train"
train(model, optimizer, EPOCHS, device, train_ds, train_loss, train_accuracy, valid_ds, val_loss, val_accuracy)

# %% [markdown] colab_type="text" id="N8m3iJgx7SV1"
# # Evaluation

# %% colab={} colab_type="code" id="HwFx4Nbh25p5"
test_imgs = []
test_labels = []

predictions = []
with tf.device(device_name=device):
    for images, labels in test_ds:
        preds = model(images)
        preds = preds.numpy()
        predictions.extend(preds)

        test_imgs.extend(images.numpy())
        test_labels.extend(labels.numpy())

# %% cellView="form" colab={} colab_type="code" id="IiutdErSpRH_"
#@title Utility functions for plotting
# Utilities for plotting

class_names = ['horse', 'human']

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = np.squeeze(img)

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    print(true_label)
  
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)



# %% cellView="form" colab={} colab_type="code" id="aVknjW4A11uz"
#@title Visualize the outputs { run: "auto" }
index = 8 #@param {type:"slider", min:0, max:9, step:1}
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(index, predictions, test_labels, test_imgs)
plt.show()

# %%
