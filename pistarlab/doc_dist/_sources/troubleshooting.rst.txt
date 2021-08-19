Troubleshooting
===============

Missing .so error when running tensorflow
-----------------------------------------
ensure LD_LIBRARY_PATH is correct


My GPU is suddenly not avilable (Ubuntu)
----------------------------------------

If your GPU was working previously, but suddenly is not accessable the the system. The following script may help the script at ```scripts/fix_gpu.sh``` may help.


Testing GPU
-----------

Check if torch is detecting the GPU


python -c "import torch; print(torch.cuda.is_available());"



Running Atari Environment on Windows
------------------------------------

If you get this error:

.. code-block:: bash

    FileNotFoundError: Could not find module '..\atari_py\ale_interface\ale_c.dll' (or one of its dependencies). Try using the full path with constructor syntax.------------

Reinstall Atari from here: https://github.com/Kojoley/atari-py/releases