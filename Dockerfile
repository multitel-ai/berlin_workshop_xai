FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /python-docker

RUN apt-get -y update
RUN apt-get -y install git

RUN apt-get -y install wget

RUN pip3 install Flask==2.2.2
RUN pip3 install gunicorn==20.1.0

RUN pip3 install ftfy regex tqdm

RUN git clone https://github.com/openai/CLIP

RUN wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -P /root/.cache/clip/

COPY visualisation/backend/ .
COPY xai_methods xai_methods

CMD ["gunicorn", "run:app", "-b", "0.0.0.0:5000"]