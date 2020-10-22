import os
import shutil
from zipfile import ZipFile
from tools import send_feedback


submission_dir = "/shared/submission/"

for file in os.listdir(submission_dir):
    if file.endswith('.zip'):
        learner_file = file
    else:
        learner_file = None

if learner_file is None:
    send_feedback(0.0, "No .zip was found in the submission directory.")
    exit()

sub_source = submission_dir + learner_file
sub_destination = '/grader/mymodel.zip'
shutil.copyfile(sub_source, sub_destination)

saved_model_path = "./mymodel.zip"

with ZipFile(saved_model_path, "r") as zipObj:
    zipObj.extractall("./")
