import ast
import types
import sys
import converter
from tools import print_stderr

# Learner Block
with open("submission/submission.py") as f:
    try:
        learner_py = ast.parse(f.read())
    except SyntaxError:
        print_stderr("The notebook you provided has a syntax error that prevents it from compiling.")
        exit()

for node in learner_py.body[:]:
    if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef)):
        learner_py.body.remove(node)

learner_module = types.ModuleType("learner_mod")
learner_code = compile(learner_py, "learner_mod.py", 'exec')
sys.modules["learner_mod"] = learner_module
exec(learner_code,  learner_module.__dict__)

# Solution Block
with open("solution/solution.py") as f:
   solution_py = ast.parse(f.read())

for node in solution_py.body[:]:
    if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.ClassDef)):
        solution_py.body.remove(node)

solution_module = types.ModuleType("solution_mod")
solution_code = compile(solution_py, "solution_mod.py", 'exec')
sys.modules["solution_mod"] = solution_module
exec(solution_code,  solution_module.__dict__)
