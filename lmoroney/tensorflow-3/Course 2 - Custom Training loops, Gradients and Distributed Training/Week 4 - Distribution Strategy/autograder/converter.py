import os
import shutil
import jupytext
from tools import send_feedback

# For production
submission_dir = "/shared/submission/"

# For Local
# submission_dir = "sub2/"

for file in os.listdir(submission_dir):
    if file.endswith('.ipynb'):
        learner_notebook = file
    else:
        learner_notebook = None

if learner_notebook is None:
    send_feedback(0.0, "No notebook was found in the submission directory.")
    exit()

sub_source = submission_dir + learner_notebook

# For production
sub_destination = '/grader/submission/submission.ipynb'

# For Local
# sub_destination = 'submission/submission.ipynb'

shutil.copyfile(sub_source, sub_destination)

nb = jupytext.read("submission/submission.ipynb")
jupytext.write(nb, 'submission/submission.py', fmt='py:percent')