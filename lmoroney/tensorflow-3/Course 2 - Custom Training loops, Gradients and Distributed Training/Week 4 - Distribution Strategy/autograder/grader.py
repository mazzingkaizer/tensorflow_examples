# +
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from tqdm import tqdm
import tensorflow_datasets as tfds
import pickle

from tensorflow.python.distribute.input_lib import DistributedDataset
from tensorflow.python.framework.ops import EagerTensor

from tensorflow import TensorSpec
import compiled
import solution_mod
import learner_mod


def get_failed_cases(test_cases):
    
    failed_cases = []
    
    for test_case in test_cases:
        name = test_case.get("name")
        got = test_case.get("got")
        expected = test_case.get("expected")

        try:
            if(type(got) == np.ndarray):
                assert np.allclose(got, expected)
            
            elif(type(got) == list):
                for a, b in zip(got, expected):
                    if(np.allclose(a,b) == False):
                        raise
            else:
                assert got == expected
            
        except:
            failed_cases.append({"name": name, "expected": expected, "got": got})
    
    return failed_cases


def Test_distribute_datasets():
    
    BATCH_SIZE_PER_REPLICA = 5
    
    strategy = tf.distribute.MirroredStrategy()
    
    # download grader_dataset from https://drive.google.com/file/d/19R6PumfmXkRtm8Pmm8tbXfQ3O80CV_hw/view?usp=sharing
    grader_dataset = tf.data.experimental.load("grader_dataset", (TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None)))
    
    train_examples = grader_dataset.take(80)
    validation_examples = grader_dataset.skip(80).take(10)
    test_examples = validation_examples.skip(10)
    
    def format_image(image, label):
        image = tf.image.resize(image, (224, 224)) / 255.0
        return  image, label
    
    train_batches = train_examples.shuffle(80 // 4).map(format_image).batch(BATCH_SIZE_PER_REPLICA).prefetch(1)
    validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE_PER_REPLICA).prefetch(1)
    test_batches = test_examples.map(format_image).batch(1)

    train_dist_dataset, val_dist_dataset, test_dist_dataset = learner_mod.distribute_datasets(strategy, train_batches, validation_batches, test_batches)
    
    if type(train_dist_dataset) != DistributedDataset:
        failed_cases = [{"name": "train_dist_dataset_type_check", 
                 "expected": DistributedDataset, 
                 "got": type(train_dist_dataset)}]
        return failed_cases, 1
    
    elif type(val_dist_dataset) != DistributedDataset:
        failed_cases = [{"name": "val_dist_dataset_type_check", 
                 "expected": DistributedDataset, 
                 "got": type(val_dist_dataset)}]
        return failed_cases, 1
    
    elif type(test_dist_dataset) != DistributedDataset:
        failed_cases = [{"name": "test_dist_dataset_type_check", 
                 "expected": DistributedDataset, 
                 "got": type(test_dist_dataset)}]
        return failed_cases, 1
    
    else:
        test_cases = [
            {
                "name": "train_dist_dataset_len_check",
                "got": len(list(train_dist_dataset)),
                "expected": len(train_batches)
            }, 
            {
                "name": "train_dist_dataset_len_check",
                "got": len(list(val_dist_dataset)),
                "expected": len(validation_batches)
            },
            {
                "name": "train_dist_dataset_len_check",
                "got": len(list(test_dist_dataset)),
                "expected": len(test_batches)
            },
        ]
        
        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)


def Test_train_test_step_fns():
    strategy = tf.distribute.MirroredStrategy()
    
    BATCH_SIZE_PER_REPLICA = 5
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
        
    grader_dataset = tf.data.experimental.load("grader_dataset", (TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None)))
    
    train_examples = grader_dataset.take(80)
    validation_examples = grader_dataset.skip(80).take(10)
    test_examples = grader_dataset.skip(90)
    
    def format_image(image, label):
        image = tf.image.resize(image, (224, 224)) / 255.0
        return  image, label
    
    train_batches = train_examples.shuffle(80 // 4).map(format_image).batch(5).prefetch(1)
    validation_batches = validation_examples.map(format_image).batch(5).prefetch(1)
    test_batches = test_examples.map(format_image).batch(1)

    train_dist_dataset, validation_dist_dataset, test_dist_dataset = learner_mod.distribute_datasets(strategy, train_batches, validation_batches, test_batches)

    # download from https://drive.google.com/file/d/1Tg4L75NeVdlNiQPK5Rfg1_qKi8eDX_E1/view?usp=sharing
    MODULE_HANDLE = 'data/resnet_50_feature_vector'

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
    
    with strategy.scope():
        
        model = ResNetModel(classes=102)
    
        optimizer = tf.keras.optimizers.Adam()
        
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
        # learner's solution
        train_step, test_step = learner_mod.train_test_step_fns(strategy, model, compute_loss, optimizer, train_accuracy, loss_object, test_loss, test_accuracy)

        if not callable(train_step):
            failed_cases = [{"name": "train_step_callable_check", 
                     "expected": True, 
                     "got": False}]
            return failed_cases, 1

        elif not callable(test_step):
            failed_cases = [{"name": "test_step_callable_check", 
                     "expected": True, 
                     "got": False}]
            return failed_cases, 1
        
        # run learner's solution from solution_mod's distributed train and test step functions
        distributed_train_step, distributed_test_step = solution_mod.distributed_train_test_step_fns(strategy, train_step, test_step, model, compute_loss, optimizer, train_accuracy, loss_object, test_loss, test_accuracy)
        
        train_result = distributed_train_step(list(train_dist_dataset)[0])        
        
        distributed_test_step(list(test_dist_dataset)[0])
        
        test_loss_result1 = test_loss.result()
        
        distributed_train_step(list(train_dist_dataset)[0])

        test_cases = [
            {
                "name": "train_result_type_check",
                "got": type(train_result),
                "expected": EagerTensor
            },
            {
                "name": "train_result_shape_check",
                "got": train_result.shape,
                "expected": ()
            },
            {
                "name": "train_result_dtype_check",
                "got": train_result.dtype,
                "expected": tf.float32
            },
            {
                "name": "test_loss_result_type_check",
                "got": type(test_loss_result1),
                "expected": EagerTensor
            },
            {
                "name": "test_loss_greater_than_zero",
                "got": test_loss_result1.numpy() > 0,
                "expected": True
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)



def Test_distributed_train_test_step_fns():
    strategy = tf.distribute.MirroredStrategy()
    
    BATCH_SIZE_PER_REPLICA = 5
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
        
    grader_dataset = tf.data.experimental.load("grader_dataset", (TensorSpec(shape=(None, None, 3), dtype=tf.uint8, name=None), TensorSpec(shape=(), dtype=tf.int64, name=None)))
    
    train_examples = grader_dataset.take(80)
    validation_examples = grader_dataset.skip(80).take(10)
    test_examples = grader_dataset.skip(90)
    
    def format_image(image, label):
        image = tf.image.resize(image, (224, 224)) / 255.0
        return  image, label
    
    train_batches = train_examples.shuffle(80 // 4).map(format_image).batch(5).prefetch(1)
    validation_batches = validation_examples.map(format_image).batch(5).prefetch(1)
    test_batches = test_examples.map(format_image).batch(1)

    train_dist_dataset, validation_dist_dataset, test_dist_dataset = learner_mod.distribute_datasets(strategy, train_batches, validation_batches, test_batches)

    MODULE_HANDLE = 'data/resnet_50_feature_vector'

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
    
    with strategy.scope():
        
        model = ResNetModel(classes=102)
    
        optimizer = tf.keras.optimizers.Adam()
        
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
        train_step, test_step = solution_mod.train_test_step_fns(strategy, model, compute_loss, optimizer, train_accuracy, loss_object, test_loss, test_accuracy)

        
        distributed_train_step, distributed_test_step = learner_mod.distributed_train_test_step_fns(strategy, train_step, test_step, model, compute_loss, optimizer, train_accuracy, loss_object, test_loss, test_accuracy)
        
        if not callable(distributed_train_step):
            failed_cases = [{"name": "distributed_train_step_callable_check", 
                     "expected": True, 
                     "got": False}]
            return failed_cases, 1

        elif not callable(distributed_test_step):
            failed_cases = [{"name": "distributed_test_step_callable_check", 
                     "expected": True, 
                     "got": False}]
            return failed_cases, 1
        
        train_result = distributed_train_step(list(train_dist_dataset)[0])        
        
        distributed_test_step(list(test_dist_dataset)[0])
        
        test_loss_result1 = test_loss.result()
        
        distributed_train_step(list(train_dist_dataset)[0])

        test_cases = [
            {
                "name": "train_result_type_check",
                "got": type(train_result),
                "expected": EagerTensor
            },
            {
                "name": "train_result_shape_check",
                "got": train_result.shape,
                "expected": ()
            },
            {
                "name": "train_result_dtype_check",
                "got": train_result.dtype,
                "expected": tf.float32
            },
            {
                "name": "test_loss_result_type_check",
                "got": type(test_loss_result1),
                "expected": EagerTensor
            },
            {
                "name": "test_loss_greater_than_zero",
                "got": test_loss_result1.numpy() > 0,
                "expected": True
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)
