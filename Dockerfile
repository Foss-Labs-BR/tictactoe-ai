#FROM tensorflow/tensorflow:2.0.0-gpu-py3-jupyter
FROM tensorflow/tensorflow:latest-gpu-jupyter

#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
#RUN apt-key del 7fa2af80
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
WORKDIR /

RUN pip install --upgrade pip
RUN pip install scikit-learn
RUN pip install keras
RUN pip install tqdm
RUN pip install imutils
RUN pip install opencv-python
RUN pip install pillow
RUN pip install tf-agents
RUN apt update
RUN apt install -y libsm6 libxext6 libxrender-dev firefox

ADD src .

#CMD python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py
CMD python main.py