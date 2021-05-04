Developer Notes
===============


Making changes to the UI
------------------------

The UI is build using Vuejs cli and requires npm to run.  Once setup, changes to the ui source code will be reflected immidiately in the browser.

#. Run the UI using ```npm run serve```
#. By default, changes will be reflected at http://localhost:8080


Building for Readonly Viewing
-----------------------------

   .. code-block:: bash
   
    pip install -e . -no-deps
    pip install -r requirements-webreadonly.txt

Building for PiPy
-----------------

1. Run Tests with tox

   .. code-block:: bash

    pip install tox
    tox

1. Building wheel and source distribution and view files

   .. code-block:: bash

    rm -rf build dist *.egg-info && 
    python setup.py bdist_wheel && python -m build --sdist --wheel && unzip -l dist/*.whl

1. Uploading to PiPy

   .. code-block:: bash

    pip install twine
    twine upload dist/*

Building the Documentation
--------------------------

1. Rebuild API Docs

    From the project root, run:

   .. code-block:: bash

    cd docs
    sphinx-apidoc -o source ../pistarlab

1. Update the HTML

    .. code-block:: bash

    cd docs
    make html
    