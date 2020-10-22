# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="jYysdyb-CaWM"
# # Custom training with tf.distribute.Strategy

# + id="dzLKpmZICaWN"
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_hub as hub

# Helper libraries
import numpy as np
import os
from tqdm import tqdm

# + [markdown] id="MM6W__qraV55"
# ## Download the dataset

# + id="cmPfSVg_lGlb"
# Note, if you get a checksum error when downloading the data
# you might need to install tfds-nightly, and then restart 
# # !pip install tfds-nightly

# + id="7NsM-Bma5wNw"
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# + id="7MqDQO0KCaWS"
splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']

(train_examples, validation_examples, test_examples), info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True, split = splits, data_dir='data/')

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

# + [markdown] id="4AXoHhrsbdF3"
# ## Create a strategy to distribute the variables and the graph

# + [markdown] id="5mVuLZhbem8d"
# How does `tf.distribute.MirroredStrategy` strategy work?
#
# *   All the variables and the model graph is replicated on the replicas.
# *   Input is evenly distributed across the replicas.
# *   Each replica calculates the loss and gradients for the input it received.
# *   The gradients are synced across all the replicas by summing them.
# *   After the sync, the same update is made to the copies of the variables on each replica.
#
# Note: You can put all the code below inside a single scope. We are dividing it into several code cells for illustration purposes.
#

# + id="F2VeZUWUj5S4"
# If the list of devices is not specified in the
# `tf.distribute.MirroredStrategy` constructor, it will be auto-detected.
strategy = tf.distribute.MirroredStrategy()

# + id="ZngeM_2o0_JO"
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# + [markdown] id="k53F5I_IiGyI"
# ## Setup input pipeline

# + [markdown] id="0Qb6nDgxiN_n"
# Export the graph and the variables to the platform-agnostic SavedModel format. After your model is saved, you can load it with or without the scope.

# + id="jwJtsCQhHK-E"
BUFFER_SIZE = num_examples

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10

# + id="rWUl3kUk8D5d"
pixels = 224
MODULE_HANDLE = 'data/resnet_50_feature_vector'
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))


# + id="RHGFit478BWD"
def format_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    return  image, label


# + [markdown] id="J7fj3GskHC8g"
# Create the datasets and distribute them:

# + id="WYrMNNDhAvVl"
train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE_PER_REPLICA).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE_PER_REPLICA).prefetch(1)
test_batches = test_examples.map(format_image).batch(1)


# -

# GRADED FUNCTION
def distribute_datasets(strategy, train_batches, validation_batches, test_batches):
    
    ### START CODE HERE ###
    train_dist_dataset = strategy.experimental_distribute_dataset(train_batches)
    val_dist_dataset = strategy.experimental_distribute_dataset(validation_batches)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_batches)
    ### END CODE HERE ###
    
    return train_dist_dataset, val_dist_dataset, test_dist_dataset



train_dist_dataset, val_dist_dataset, test_dist_dataset = distribute_datasets(strategy, train_batches, validation_batches, test_batches)

type(train_dist_dataset)


# + [markdown] id="bAXAo_wWbWSb"
# ## Create the model
#
# We use the Model Subclassing API to do this.

# + id="9ODch-OFCaW4"
class ResNetModel(tf.keras.Model):
    def __init__(self, classes):
        super(ResNetModel, self).__init__()
        self._feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                                 trainable=False) 
        self._classifier = tf.keras.layers.Dense(classes, activation='softmax')

    def call(self, inputs):
        x = self._feature_extractor(inputs)
        x = self._classifier(x)
        return x


# + id="9iagoTBfijUz"
# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# + [markdown] id="e-wlFFZbP33n"
# ## Define the loss function
#

# + id="R144Wci782ix"
with strategy.scope():
    # Set reduction to `none` so we can do the reduction afterwards and divide by
    # global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    # or loss_fn = tf.keras.losses.sparse_categorical_crossentropy
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    test_loss = tf.keras.metrics.Mean(name='test_loss')
# -



# + [markdown] id="w8y54-o9T2Ni"
# ## Define the metrics to track loss and accuracy
#
# These metrics track the test loss and training and test accuracy. You can use `.result()` to get the accumulated statistics at any time.

# + id="zt3AHb46Tr3w"
with strategy.scope():
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

# + [markdown] id="iuKuNXPORfqJ"
# ## Training loop

# + id="OrMmakq5EqeQ"
# model and optimizer must be created under `strategy.scope`.
with strategy.scope():
    model = ResNetModel(classes=num_classes)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


# + id="zUQ_nAP1MtA9"
# GRADED FUNCTION
def train_test_step_fns(strategy, model, compute_loss, optimizer, train_accuracy, loss_object, test_loss, test_accuracy):
    with strategy.scope():
        def train_step(inputs):
            images, labels = inputs

            with tf.GradientTape() as tape:
                ### START CODE HERE ###
                predictions = model(images, training=True) # @REPLACE predictions = model(None, None)
                loss = compute_loss(labels, predictions) # @REPLACE loss = compute_loss(None, None)
                ### END CODE HERE ###

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_accuracy.update_state(labels, predictions)
            return loss 

        def test_step(inputs):
            images, labels = inputs
            
            ### START CODE HERE ###
            predictions = model(images, training=False) # @REPLACE predictions = model(None, None)
            t_loss = loss_object(labels, predictions) # @REPLACE t_loss = loss_object(None, None)
            ### END CODE HERE ###

            test_loss.update_state(t_loss)
            test_accuracy.update_state(labels, predictions)
        
        return train_step, test_step


# -

train_step, test_step = train_test_step_fns(strategy, model, compute_loss, optimizer, train_accuracy, loss_object, test_loss, test_accuracy)


def distributed_train_test_step_fns(strategy, train_step, test_step, model, compute_loss, optimizer, train_accuracy, loss_object, test_loss, test_accuracy):
    with strategy.scope():
        @tf.function
        def distributed_train_step(dataset_inputs):
            ### START CODE HERE ###
            per_replica_losses = strategy.run(train_step, args=(dataset_inputs,)) # @REPLACE per_replica_losses = strategy.run(None, None)
            ### END CODE HERE ###
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            ### START CODE HERE ###
            return strategy.run(test_step, args=(dataset_inputs,)) # @REPLACE return strategy.run(None, None)
            ### END CODE HERE ###
    
        return distributed_train_step, distributed_test_step


distributed_train_step, distributed_test_step = distributed_train_test_step_fns(strategy, train_step, test_step, model, compute_loss, optimizer, train_accuracy, loss_object, test_loss, test_accuracy)

# + id="gX975dMSNw0e"
with strategy.scope():
    for epoch in range(EPOCHS):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in tqdm(train_dist_dataset):
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        # TEST LOOP
        for x in test_dist_dataset:
            distributed_test_step(x)

        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                    "Test Accuracy: {}")
        print (template.format(epoch+1, train_loss,
                               train_accuracy.result()*100, test_loss.result(),
                               test_accuracy.result()*100))

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

# + [markdown] id="Z1YvXqOpwy08"
# Things to note in the example above:
#
# * We are iterating over the `train_dist_dataset` and `test_dist_dataset` using  a `for x in ...` construct.
# * The scaled loss is the return value of the `distributed_train_step`. This value is aggregated across replicas using the `tf.distribute.Strategy.reduce` call and then across batches by summing the return value of the `tf.distribute.Strategy.reduce` calls.
# * `tf.keras.Metrics` should be updated inside `train_step` and `test_step` that gets executed by `tf.distribute.Strategy.experimental_run_v2`.
# *`tf.distribute.Strategy.experimental_run_v2` returns results from each local replica in the strategy, and there are multiple ways to consume this result. You can do `tf.distribute.Strategy.reduce` to get an aggregated value. You can also do `tf.distribute.Strategy.experimental_local_results` to get the list of values contained in the result, one per local replica.
#

# + [markdown] id="WEaNCzYQvFqo"
# # Save the Model as a SavedModel
#
# You'll get a saved model of this finished trained model. You'll then 
# need to zip that up to upload it to the testing infrastructure. We
# provide the code to help you with that here
#
# ## Step 1: Save the model as a SavedModel
# This code will save your model as a SavedModel

# + id="1zAlTlRxrqFu"
model_save_path = "./tmp/mymodel/1/"
tf.saved_model.save(model, model_save_path)

# + [markdown] id="e0Zfmx6LvTJA"
# ## Step 2: Zip the SavedModel Directory into /mymodel.zip
#
# This code will zip your saved model directory contents into a single file.
# You can use the file browser pane to the left of colab to find mymodel.zip
# Right click on it and select 'Download'. It's a large file, so it might
# take some time.
#
# If the download fails because you aren't allowed to download multiple files from colab, check out the guidance here: https://ccm.net/faq/32938-google-chrome-allow-websites-to-perform-simultaneous-downloads

# + id="gMuo2wQls41l"
import os
import zipfile

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

zipf = zipfile.ZipFile('./mymodel.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('./tmp/mymodel/1/', zipf)
zipf.close()
