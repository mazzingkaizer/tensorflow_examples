import sys
from disable_warnings import *
from tools import print_stderr, send_feedback
from grader import (Test_tf_constant, Test_tf_square,
                    Test_tf_reshape, Test_tf_cast, Test_tf_multiply, 
                    Test_tf_add, Test_tf_gradient_tape)


def run_grader(part_id):
    
    graded_funcs = {
        "r9hUX": Test_tf_constant, 
        "kWeLo": Test_tf_square,
        "E1weD": Test_tf_reshape,
        "AuDvh": Test_tf_cast,
        "NI9Co": Test_tf_multiply,
        "Yz2KA": Test_tf_add,
        "4uBzv": Test_tf_gradient_tape
    }

    g_func = graded_funcs.get(part_id)
    if g_func is None:
        print_stderr("The partID provided did not correspond to any graded function.")
        return

    failed_cases, num_cases = g_func()
    score = 1.0 - len(failed_cases) / num_cases
    if failed_cases:
        failed_msg = ""
        for failed_case in failed_cases:
            failed_msg += f"Failed {failed_case.get('name')}.\nExpected:\n{failed_case.get('expected')},\nbut got:\n{failed_case.get('got')}.\n\n"
        
        send_feedback(score, failed_msg)
    else:
        send_feedback(score, "All tests passed! Congratulations!")
    

if __name__ == "__main__":
    try:
        part_id = sys.argv[2]
    except IndexError:
        print_stderr("Missing partId. Required to continue.")
        send_feedback(0.0, "Missing partId.")
    else:
        run_grader(part_id)
