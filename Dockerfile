FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /python-docker

RUN apt-get -y update
RUN apt-get -y install git

RUN pip3 install Flask==2.2.2
RUN pip3 install gunicorn==20.1.0

RUN pip3 install ftfy regex tqdm

COPY visualisation/backend/ .
COPY xai_methods xai_methods

RUN git clone https://github.com/openai/CLIP

CMD ["gunicorn", "run:app", "-b", "0.0.0.0:5000"]