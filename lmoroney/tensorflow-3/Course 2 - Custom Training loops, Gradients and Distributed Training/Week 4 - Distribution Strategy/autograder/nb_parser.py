import json
from tools import print_stderr

def get_graded_cells():
    with open("submission/submission.ipynb") as f:
        nb_json = json.loads(f.read())

    cells = {}
    for cell in nb_json.get('cells'):
        metadata = cell.get('metadata')
        if metadata.get('graded'):
            cells.update({metadata.get('name'): cell})
    return cells


def cell_output_pattern_matching(cell, patterns):
    if cell.get('cell_type') != 'code':
        err_msg = "evaluated cell is not a code cell."
        print_stderr(err_msg)
        return False, err_msg

    output_dict = cell.get('outputs')[0]
    if output_dict.get('ename'):
        return False, 'cell has errors.'

    if output_dict.get('name'):
        output_txt = output_dict.get('text')
        
        for line in output_txt:
            missing_patterns = [k for k,v in patterns.items() if not v]
            for mp in missing_patterns:
                if mp in line:
                    patterns.update({mp: True})

        if not all(patterns.values()):
            return False, "missing patterns."

        return True, "passed!"
        

# [{'ename': 'SyntaxError', 'evalue': 'invalid syntax (<ipython-input-3-cfa0d70b7e6c>, line 2)', 'output_type': 'error', 'traceback': ['\x1b[0;36m  File \x1b[0;32m"<ipython-input-3-cfa0d70b7e6c>"\x1b[0;36m, line \x1b[0;32m2\x1b[0m\n\x1b[0;31m    a =& b\x1b[0m\n\x1b[0m       ^\x1b[0m\n\x1b[0;31mSyntaxError\x1b[0m\x1b[0;31m:\x1b[0m invalid syntax\n']}]
# Examples of how cell's output look depending on error
# NO ERROR:
# [{'name': 'stdout', 'output_type': 'stream', 'text': ['\n', 'Step      1: Ran 1 train steps in 8.33 secs\n', 'Step      1: train CrossEntropyLoss |  10.41197491\n', 'Step      1: eval  CrossEntropyLoss |  10.41386223\n', 'Step      1: eval          Accuracy |  0.00000000\n', '\n', 'Step     10: Ran 9 train steps in 64.94 secs\n', 'Step     10: train CrossEntropyLoss |  10.41337585\n', 'Step     10: eval  CrossEntropyLoss |  10.41288471\n', 'Step     10: eval          Accuracy |  0.00000000\n']}]

# ERROR
# [{'output_type': 'error', 'ename': 'SyntaxError', 'evalue': 'invalid syntax (<ipython-input-3-cfa0d70b7e6c>, line 2)', 'traceback': ['\x1b[0;36m  File \x1b[0;32m"<ipython-input-3-cfa0d70b7e6c>"\x1b[0;36m, line \x1b[0;32m2\x1b[0m\n\x1b[0;31m    a =& b\x1b[0m\n\x1b[0m       ^\x1b[0m\n\x1b[0;31mSyntaxError\x1b[0m\x1b[0;31m:\x1b[0m invalid syntax\n']}]
