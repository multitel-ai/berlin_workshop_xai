# berlin_workshop_xai
TRAIL’22 Workshop at Berlin   -  Explore the gap between existing explainable artificial intelligence techniques and their use as tools

## Introduction

This repository contains an API that serves a [CLIP][1] model. The vanilla CLIP model is modified so that a state of the art explainability method can be applied to it. The method we applied is based on Chefer's [work][2] on XAI for multimodal transformers.

## Usage

The CLIP API can be launched in a docker container. Simply run 

```bash
docker-compose up --build
```

This will launck a flask app that can be contacted at port `5000` of your local machine. A full example of how to interact with the API is given in the `api_demo.ipynb` notebook. To run the notebook, run a `jupyter` server in the running container with:

```bash
docker exec YOUR_CONTAINER_ID jupyter notebook --allow-root --ip=0.0.0.0
```
Copy the URL logged in the console and you should have acces to a notebook running in the container.

You can find your docker container id with
```bash
docker ps
```

Alternatively, you can test the API on your host in `python` with:

```python
import requests

endpoint = 'http://localhost:5000/attentions'

img_url = 'https://pictures-of-cats.org/wp-content/uploads/2018/02/mouse-and-cat-x.jpg'
text = 'A mouse'

def get_attentions_from_api(endpoint, img_url, text):
    headers = {'Content-Type': 'application/json'}

    data = {
        'img_url': img_url,
        'text': text
    }
    
    response = requests.post(endpoint, headers=headers, json=data)
    
    response_headers = response.headers
    response_body = response.json()
    
    return response_body

response_dict = get_attentions_from_api(endpoint, img_url, text)
```

### Docker installation

In order to run the CLIP API, these prerequisites should be met:

* A NVIDIA driver should be installed
* Docker should be installed. To install docker, follow the instructions detailed [here](https://docs.docker.com/engine/install/ubuntu/). 
* docker-compose should be installed (only if you plan to use the pipeline in dev mode). To install docker-compose, follow the instructions [here](https://docs.docker.com/compose/install/) (in the Linux tab).
* GPU support should be installed for docker. Follow instuctions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).


## SOA

During the workshop, a [SOA repository][3] was created. There, blogposts, papers, useful tools etc. are listed.

## AudioCLIP

In future works, it was planned to use AudioCLIP to incorportate the audio modality on top of image and text modalities. We created a docker environment to run that model in the `AudioCLIP` folder.

Original AudioCLIP repo: https://github.com/AndreyGuzhov/AudioCLIP

Paper: https://arxiv.org/pdf/2106.13043v1.pdf

Installation instructions are given in the `AudioCLIP` folder.


[1]: <https://github.com/openai/CLIP> "CLIP repository"
[2]: <https://github.com/hila-chefer/Transformer-MM-Explainability> "Chefer's repository"
[3]: <https://github.com/aenglebert/Transformer_XAI> "SOA repository"
