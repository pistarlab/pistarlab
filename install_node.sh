#!/usr/bin/env bash
sudo apt-get install -y nodejs npm && sudo npm cache clean -f && sudo npm install -g n && sudo n 12.14.1
sudo npm install --global yarn@1.7.0

# NOTE: if having node conflict, you may want to reset your apt source repositories 