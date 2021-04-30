#!/bin/bash
kill -09 $(nvidia-smi | awk '$2=="Processes:" {p=1} p && $2 == 0 && $3 > 0 {print $3}')
python -c "import torch; print(torch.cuda.is_available());"
sudo systemctl stop systemd-logind
sleep 1
sudo systemctl stop systemd-logind
sleep 1
sudo systemctl stop systemd-logind
sleep 1
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset 
sudo rmmod nvidia_uvm 
sudo rmmod nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia
sudo nvidia-smi -r
sudo nvidia-smi
python -c "import torch; print(torch.cuda.is_available());"
sudo systemctl start systemd-logind
