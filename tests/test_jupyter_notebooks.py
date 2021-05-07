#https://nbconvert.readthedocs.io/en/latest/execute_api.html

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

def run_test_notebook(notebook_filename,output_path):
    with open(os.path.join('tests',notebook_filename),'r') as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(
        timeout=600, 
        kernel_name='python'
        )
    ep.preprocess(nb)

    os.makedirs(output_path,exist_ok=True)
    with open(os.path.join(output_path,notebook_filename), 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

test_notebook_output_path = os.path.join('tests','notebook_output')   

class TestJupyterNotebooks:

    def test_nbs(self):
        run_test_notebook("test_nb_example.ipynb",test_notebook_output_path)
