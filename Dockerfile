FROM rayproject/ray:1.2.0-gpu

# Needed to run Tune example with a 'plot' call - which does not actually render a plot, but throws an error.
RUN sudo apt-get update && sudo apt-get install -y zlib1g-dev libgl1-mesa-dev libgtk2.0-dev python-opengl xvfb ffmpeg && sudo apt-get clean

ENV PATH /usr/local/bin:$PATH

# PYTHONDONTWRITEBYTECODE: Keeps Python from generating .pyc files in the container
# PYTHONUNBUFFERED: Turns off buffering for easier container logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 
RUN env

RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# RUN sudo apt-get install -y libcudnn7=7.6.5.32-1+cuda10.2
RUN whoami
RUN sudo apt-get install -y nodejs npm && sudo npm cache clean -f && sudo npm install -g n && sudo n 12.14.1
RUN sudo npm install --global yarn@1.7.0
RUN pip install --no-cache-dir -U pip \
    tensorflow==2.3.1
COPY ./requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
RUN sudo chown -R ray:users /home/ray/.config/
COPY --chown=ray:users . /app
WORKDIR /app
RUN pip install -e . --no-deps
ENV PYTHONPATH "${PYTHONPATH}:/app"

# TODO: not needed for ray workers
RUN cd /app/ && ./build_redis.sh
RUN cd /app/ && ./build_ui.sh
# RUN cd /app/ && ./build_ide.sh

EXPOSE 8080 
EXPOSE 7777 
EXPOSE 7776 
EXPOSE 7778 
EXPOSE 7781 
EXPOSE 8265
ENV PYTHONUSERBASE /home/ray/pistarlab/plugins/site-packages/
ENTRYPOINT "pistarlab_launcher"
