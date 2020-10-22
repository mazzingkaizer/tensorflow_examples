import sys
from disable_warnings import *
from tools import print_stderr, send_feedback
from grader import (Test_F1ScoreResult, Test_apply_gradients, 
                    Test_train_data_for_one_epoch)



def run_grader(part_id):
    
    graded_funcs = {
        "mylyu": Test_F1ScoreResult, 
        "YQutN": Test_apply_gradients,
        "8dNmL": Test_train_data_for_one_epoch
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
