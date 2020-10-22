import sys
import json
import numpy

def send_feedback(score, msg):
    post = {"fractionalScore": score, "feedback": msg}
    print(json.dumps(post))

def print_stderr(error_msg):
    print(str(error_msg), file=sys.stderr)