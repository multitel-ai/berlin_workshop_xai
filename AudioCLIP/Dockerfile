FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt-get update
RUN apt-get install wget -y

RUN apt-get install libsndfile1 -y

RUN pip install numpy==1.18.1
RUN pip install pandas==1.0.3
RUN pip install pytorch-ignite==0.3.0
RUN pip install scikit-learn==0.22.1
RUN pip install scipy==1.4.1
RUN pip install termcolor==1.1.0
RUN pip install torchvision==0.10.0
RUN pip install tqdm==4.43.0
RUN pip install visdom==0.1.8.9
RUN pip install numba==0.48
RUN pip install librosa==0.9.2
RUN pip install matplotlib==3.5.3
RUN pip install simplejpeg==1.6.5
RUN pip install ftfy==6.1.1
RUN pip install regex==2022.8.17

RUN pip install jupyter

# TODO: later once requrements are set, put all this in a requirements file
# COPY requirements.txt /tmp/requirements.txt
# RUN pip install -r /tmp/requirements.txt

ARG GROUP_ID
ARG USER_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user

# RUN mkdir /notebook_data
WORKDIR /notebook_data

CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --NotebookApp.password=sha1:42538b15bee4:ca34a87eeb8a877639b50350ede11b2a3607c0f6