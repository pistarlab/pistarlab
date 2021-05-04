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

```bash
python -c "import torch; print(torch.cuda.is_available());"
```


