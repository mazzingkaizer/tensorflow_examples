import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

import numpy as np
import itertools
from tqdm import tqdm
import tensorflow_datasets as tfds

import pickle
from tensorflow.python.framework.ops import EagerTensor

import compiled
import learner_mod
import solution_mod


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


def Test_F1ScoreResult():
        
    def get_result(F1Score_class, tp, fp, tn, fn):
        F1Score_class.tp = tf.Variable(tp, dtype = 'int32')
        F1Score_class.fp = tf.Variable(fp, dtype = 'int32')
        F1Score_class.tn = tf.Variable(tn, dtype = 'int32')
        F1Score_class.fn = tf.Variable(fn, dtype = 'int32')
        
        return F1Score_class.result()
    
    learner_F1Score = learner_mod.F1Score()
    solution_F1Score = solution_mod.F1Score()
    
    result1 = get_result(learner_F1Score, 2, 5, 7, 9)
    expected1 = get_result(solution_F1Score, 2, 5, 7, 9)
    
    result2 = get_result(learner_F1Score, 8, 16, 24, 32)
    expected2 = get_result(solution_F1Score, 8, 16, 24, 32)

    if type(result1) != EagerTensor:
        failed_cases = [{"name": "type_check", 
                         "expected": EagerTensor, 
                         "got": type(result1)}]
        
        return failed_cases, 1
    
    else:
        test_cases = [
            {
                "name": "type_check",
                "got": type(result1),
                "expected": EagerTensor,
                "error_message": 'result has an incorrect type.'
            },
            {
                "name": "shape_check",
                "got": result1.shape,
                "expected": (),
                "error_message": "output shape is incorrect"
            },
            {
                "name": "dtype_check",
                "got": result1.dtype,
                "expected": np.float64,
                "error_message": "output dtype is incorrect"
            },
            {
                "name": "output_check1",
                "got": result1,
                "expected": expected1,
                "error_message": f'output of the hidden test is {expected1} while you got {result1}'
            },
            {
                "name": "output_check2",
                "got": result2,
                "expected": expected2,
                "error_message": f'output of the hidden test is {expected2} while you got {result2}'
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)


def Test_apply_gradients():

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    loss_object = tf.keras.losses.BinaryCrossentropy()

    def base_model():
        inputs = tf.keras.layers.Input(shape=(9))

        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    model = base_model()
    solution_model = base_model()
    
    with open('train_batch.pkl', 'rb') as f:
        (x_test, y_test) = pickle.load(f)
    
    with open('model_weights.pkl', 'rb') as f:
        model_weights = pickle.load(f)

    model.set_weights(model_weights)
    solution_model.set_weights(model_weights)
    
    result = learner_mod.apply_gradient(optimizer, loss_object, model, x_test, y_test)
    expected = solution_mod.apply_gradient(optimizer, loss_object, solution_model, x_test, y_test)
        
    with open('expected_weights.pkl', 'rb') as f:
        expected_weights = pickle.load(f)
        
    if (type(result[0]) != EagerTensor):
        failed_cases = [{"name": "logits_type_check", 
                         "expected": EagerTensor, 
                         "got": type(result[0])}]
        
        return failed_cases, 1
    
    elif (type(result[1]) != EagerTensor):
        failed_cases = [{"name": "loss_type_check", 
                         "expected": EagerTensor, 
                         "got": type(result[1])}]
        
        return failed_cases, 1
    
    else:
        test_cases = [
            {
                "name": "logits_shape_check",
                "got": result[0].shape,
                "expected": expected[0].shape,
                "error_message": "output shape is incorrect"
            },
            {
                "name": "logits_dtype_check",
                "got": result[0].dtype,
                "expected": expected[0].dtype,
                "error_message": "output dtype is incorrect"
            },
            {
                "name": "logits_output_array_check",
                "got": result[0].numpy(),
                "expected": expected[0].numpy(),
                "error_message": "output array is incorrect"
            },
            {
                "name": "loss_shape_check",
                "got": result[1].shape,
                "expected": expected[1].shape,
                "error_message": "output shape is incorrect"
            },
            {
                "name": "loss_dtype_check",
                "got": result[1].dtype,
                "expected": expected[1].dtype,
                "error_message": "output dtype is incorrect"
            },
            {
                "name": "loss_output_check",
                "got": result[1].numpy(),
                "expected": expected[1].numpy(),
                "error_message": "output is incorrect"
            },
            {
                "name": "weights_check",
                "got": model.get_weights(),
                "expected": expected_weights,
                "error_message": "output array is incorrect"
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)


def Test_train_data_for_one_epoch():

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    loss_object = tf.keras.losses.BinaryCrossentropy()
    learner_train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    solution_train_acc_metric = tf.keras.metrics.BinaryAccuracy()
    learner_train_f1score_metric = learner_mod.F1Score()
    solution_train_f1score_metric = learner_mod.F1Score()

    def base_model():
        inputs = tf.keras.layers.Input(shape=(9))

        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    model = base_model()
    solution_model = base_model()
    
    with open('train_batch.pkl', 'rb') as f:
        (x_test, y_test) = pickle.load(f)
    
    with open('model_weights.pkl', 'rb') as f:
        model_weights = pickle.load(f)

    model.set_weights(model_weights)
    solution_model.set_weights(model_weights)
    
    with open('train_dataset.pkl', 'rb') as f:
        x, y = pickle.load(f)

    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.batch(32)
    
    result = learner_mod.train_data_for_one_epoch(train_dataset, optimizer, loss_object, model, learner_train_acc_metric, learner_train_f1score_metric, verbose=False)
    expected = solution_mod.train_data_for_one_epoch(train_dataset, optimizer, loss_object, solution_model, solution_train_acc_metric, solution_train_f1score_metric, verbose=False)   
    
    if (type(result) != list):
        failed_cases = [{"name": "logits_type_check", 
                         "expected": EagerTensor, 
                         "got": type(result[0])}]
        
        return failed_cases, 1
    
    else:
        test_cases = [
            {
                "name": "result_length_check",
                "got": len(result),
                "expected": len(expected),
                "error_message": "output shape is incorrect"
            },
            {
                "name": "losses_output_check",
                "got": result,
                "expected": expected,
                "error_message": "output is incorrect"
            },
            {
                "name": "train_acc_metric_check",
                "got": learner_train_acc_metric.result().numpy(),
                "expected": solution_train_acc_metric.result().numpy(),
                "error_message": "binary accuracy metric is not updated"
            },
            {
                "name": "train_f1_score_metric_check",
                "got": learner_train_f1score_metric.result().numpy(),
                "expected": solution_train_f1score_metric.result().numpy(),
                "error_message": "F1 Score metric is not updated"
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)