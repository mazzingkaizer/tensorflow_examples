import tensorflow as tf
import numpy as np
from tensorflow.python.framework.ops import EagerTensor
import compiled
import learner_mod
import solution_mod
from tools import send_feedback

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


def Test_tf_constant():

    x = np.arange(41, 50)
    y = np.arange(71, 80)
    
    target = learner_mod.tf_constant
    solution = tf.constant
    
    result1 = target(x)
    result2 = target(y)

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
                "expected": (len(x),),
                "error_message": "output shape is incorrect"
            },
            {
                "name": "dtype_check",
                "got": result1.dtype,
                "expected": np.int64,
                "error_message": "output dtype is incorrect"
            },
            {
                "name": "hidden_input_array_check_1",
                "got": result1.numpy(),
                "expected": x,
                "error_message": f'output of the hidden test is {result1.numpy()} while you got {x}'
            },
            {
                "name": "hidden_input_array_check_2",
                "got": result2.numpy(),
                "expected": y,
                "error_message": f'output of the hidden test is {result2.numpy()} while you got {y}'
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)


def Test_tf_square():

    x = np.arange(41, 50)
    y = np.arange(71, 80)
    
    target = learner_mod.tf_square
    solution = tf.square
    
    result1 = target(x)
    result2 = target(y)

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
                "expected": (len(x),),
                "error_message": "output shape is incorrect"
            },
            {
                "name": "dtype_check",
                "got": result1.dtype,
                "expected": np.int64,
                "error_message": "output dtype is incorrect"
            },
            {
                "name": "hidden_input_array_check_1",
                "got": result1.numpy(),
                "expected": solution(x).numpy(),
                "error_message": "output array is incorrect"
            },
            {
                "name": "hidden_input_array_check_2",
                "got": result2.numpy(),
                "expected": solution(y).numpy(),
                "error_message": "output array is incorrect"
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)


def Test_tf_reshape():

    x = np.arange(41, 57)
    y = np.arange(71, 87)
    
    target = learner_mod.tf_reshape
    solution = tf.reshape
    
    shape = (4, 4)
    
    result1 = target(x, shape)
    result2 = target(y, shape)

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
                "expected": shape,
                "error_message": "output shape is incorrect"
            },
            {
                "name": "dtype_check",
                "got": result1.dtype,
                "expected": np.int64,
                "error_message": "output dtype is incorrect"
            },
            {
                "name": "hidden_input_array_check_1",
                "got": result1.numpy(),
                "expected": solution(x, shape).numpy(),
                "error_message": "output array is incorrect"
            },
            {
                "name": "hidden_input_array_check_2",
                "got": result2.numpy(),
                "expected": solution(y, shape).numpy(),
                "error_message": "output array is incorrect"
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)


def Test_tf_cast():

    x = np.arange(41, 50)
    y = np.arange(71, 80)
    
    target = learner_mod.tf_cast
    solution = tf.cast
    
    test_dtype = tf.uint32
    
    result1 = target(x, test_dtype)
    result2 = target(y, test_dtype)

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
                "expected": (len(x),),
                "error_message": "output shape is incorrect"
            },
            {
                "name": "dtype_check",
                "got": result1.dtype,
                "expected": test_dtype,
                "error_message": "output dtype is incorrect"
            },
            {
                "name": "hidden_input_array_check_1",
                "got": result1.numpy(),
                "expected": solution(x, test_dtype).numpy(),
                "error_message": "output array is incorrect"
            },
            {
                "name": "hidden_input_array_check_2",
                "got": result2.numpy(),
                "expected": solution(y, test_dtype).numpy(),
                "error_message": "output array is incorrect"
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)


def Test_tf_multiply():

    x1 = np.arange(41, 50)
    y1 = np.arange(71, 80)
    x2 = np.arange(31, 40)
    y2 = np.arange(11, 20)
    
    target = learner_mod.tf_multiply
    solution = tf.multiply
    
    result1 = target(x1, y1)
    result2 = target(x2, y2)

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
                "expected": (len(x1),),
                "error_message": "output shape is incorrect"
            },
            {
                "name": "dtype_check",
                "got": result1.dtype,
                "expected": np.int64,
                "error_message": "output dtype is incorrect"
            },
            {
                "name": "hidden_input_array_check_1",
                "got": result1.numpy(),
                "expected": solution(x1, y1),
                "error_message": "output array is incorrect"
            },
            {
                "name": "hidden_input_array_check_2",
                "got": result2.numpy(),
                "expected": solution(x2, y2),
                "error_message": "output array is incorrect"
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)


def Test_tf_add():

    x1 = tf.constant(np.arange(41, 50))
    y1 = tf.constant(np.arange(71, 80))
    x2 = tf.constant(np.arange(31, 40))
    y2 = tf.constant(np.arange(11, 20))
    
    target = learner_mod.tf_add
    solution = tf.add
    
    result1 = target(x1, y1)
    result2 = target(x2, y2)

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
                "expected": (len(x1),),
                "error_message": "output shape is incorrect"
            },
            {
                "name": "dtype_check",
                "got": result1.dtype,
                "expected": np.int64,
                "error_message": "output dtype is incorrect"
            },
            {
                "name": "hidden_input_array_check_1",
                "got": result1.numpy(),
                "expected": solution(x1, y1),
                "error_message": "output array is incorrect"
            },
            {
                "name": "hidden_input_array_check_2",
                "got": result2.numpy(),
                "expected": solution(x2, y2),
                "error_message": "output array is incorrect"
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)


def Test_tf_gradient_tape():

    x = tf.constant(4.0)
    y = tf.constant(8.0)
    
    target = learner_mod.tf_gradient_tape
    
    def solution(x):
        with tf.GradientTape() as t:

            t.watch(x) 

            y =  3 * (x ** 3) - 2 * (x ** 2)  + x    

            z = tf.reduce_sum(y) 

        dz_dx = t.gradient(z, x)

        return dz_dx
    
    result1 = target(x)
    result2 = target(y)

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
                "expected": tf.float32,
                "error_message": "output dtype is incorrect"
            },
            {
                "name": "hidden_input_check_1",
                "got": result1.numpy(),
                "expected": solution(x).numpy(),
                "error_message": "output array is incorrect"
            },
            {
                "name": "hidden_input_check_2",
                "got": result2.numpy(),
                "expected": solution(y).numpy(),
                "error_message": "output array is incorrect"
            },
        ]

        failed_cases = get_failed_cases(test_cases)

        return failed_cases, len(test_cases)
