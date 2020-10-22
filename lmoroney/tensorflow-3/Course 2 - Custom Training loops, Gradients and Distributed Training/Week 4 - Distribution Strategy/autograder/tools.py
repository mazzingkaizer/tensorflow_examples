import sys
import json
import numpy


def Equality(out, exp):
    """Checks for equality. Needed when dealing with types that have ambiguous equality.

    Args:
        out (Object): Output of learner's version.
        exp (Object): Expected output.

    Returns:
        bool: True if the two objects are equal, False otherwise.
    """
    assert type(out) == type(exp), "incompatible types for comparison"
    if isinstance(out, (numpy.ndarray)):
        return (out == exp).all()
    return out == exp


def table_testing_assert(func, test_cases, str_rep=False, assert_func=Equality):
    """Tests the execution of a graded function.

    Args:
        func (function): Graded function.
        test_cases (list): List of cases in dict format.
        str_rep (bool, optional): True if the str representations are going to be compared. Defaults to False.
        assert_func (function, optional): Assert function. Defaults to Equality.
    Returns:
        score: Score received.
    """
    failed_test_dict = []
    for test_case in test_cases:
        name = test_case.get("name")
        input_dict = test_case.get("input")
        expected = test_case.get("expected")
        if None in (name, input_dict, expected):
            print("malformed test case")
            return

        output = func(**input_dict)
        if str_rep:
            expected = str(expected)
            output = str(output)

        if not assert_func(output, expected):
            failed_test_dict.append({"name": name, "expected": expected, "got": output})

    return failed_test_dict


def send_feedback(score, msg):
    post = {"fractionalScore": score, "feedback": msg}
    print(json.dumps(post))

def print_stderr(error_msg):
    print(str(error_msg), file=sys.stderr)