#FROM stereolabs/zed:4.0-gl-devel-cuda11.4-ubuntu20.04
FROM stereolabs/zed:4.2-gl-devel-cuda11.4-ubuntu20.04
RUN apt-get update
RUN apt-get install -y python3
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt
RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR "/usr/local/zed"
RUN python3 get_python_api.py
RUN apt-get install -y usbutils neovim

RUN useradd -u 1000 -g 1000 -m d_phil
#USER d_phil
#USER root
