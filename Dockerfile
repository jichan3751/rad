FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /workspace

# install apt-get packages
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    build-essential \
    gnupg2 \
    make \
    cmake \
    ffmpeg \
    swig \
    libz-dev \
    unzip \
    zlib1g-dev \
    libglfw3 \
    libglfw3-dev \
    libxrandr2 \
    libxinerama-dev \
    libxi6 \
    libxcursor-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    lsb-release \
    ack-grep \
    patchelf \
    wget \
    xpra \
    xserver-xorg-dev \
    xvfb \
    python-opengl \
    ffmpeg \
    libmagickwand-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install mujoco
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco.zip
RUN ln -s /root/.mujoco/mujoco200 /root/.mujoco/mujoco200_linux
COPY ./mjkey.txt /root/.mujoco/
ENV MJC_PATH /root/.mujoco
ENV MJLIB_PATH /root/.mujoco/mujoco200/bin/libmujoco200.so
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
RUN cd /root/.mujoco/ \
    && git clone https://github.com/openai/mujoco-py.git \
    && cd mujoco-py \
    && pip install -e . \
    && python -c "import mujoco_py"

# install pip packages
RUN pip install \
    ipdb \
    gym \
    matplotlib \
    ipython \
    moviepy \
    torch \
    torchvision \
    opencv-python \
    pillow \
    box2d-py \
    termcolor \
    tb-nightly \
    absl-py \
    pyparsing \
    imageio \
    imageio-ffmpeg \
    scikit-image \
    tabulate \
    pyvirtualdisplay==1.3.2 \
    Wand

# install pip packages 2
RUN pip install \
    git+git://github.com/deepmind/dm_control.git \
    git+git://github.com/1nadequacy/dmc2gym.git

