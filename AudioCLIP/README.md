# AudioClip

Original repo: https://github.com/AndreyGuzhov/AudioCLIP

Paper: https://arxiv.org/pdf/2106.13043v1.pdf

## Installation

To install AudioCLIP, you can either use a venv, or use a Docker image.

### venv

TODO

### Docker

This approach requires:
* A NVIDIA driver installed
* Docker should be installed. To install docker, follow the instructions detailed [here](https://docs.docker.com/engine/install/ubuntu/). 
* GPU support should be installed for docker. Follow instuctions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
* Optionnal: docker-compose To install docker-compose, follow the instructions [here](https://docs.docker.com/compose/install/) (in the Linux tab).

In the `docker-compose.yml` file, you should adjust volumes to mount yout local folder.

- First clone the AudioCLIP repo (in the `berlin_workshop_xai/AudioCLIP` folder):
``` bash
git clone https://github.com/AndreyGuzhov/AudioCLIP
```
The repo uses GitLFS with a low quota so if you get errors downloading the weights just ignore it and download them manually from [here](https://github.com/AndreyGuzhov/AudioCLIP/releases), or with the CLI:

``` bash
wget -P AudioCLIP/assets/ https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt
```
Weights should be placed in the `assets` folder

- You can buid and launch the image with
``` bash
USER_ID=$(id -u) GROUP_ID=$(id -g) docker-compose up --build
```

On subsequent launches, ignore the `--build` flag when you don't need to rebuild the image.

Alternatively, you can use plain Docker to build and launch the image:

``` bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t  audioclip .
```

``` bash
docker run -p 7799:8888 -v /home/imagedpt/Desktop/AudioCLIP:/notebook_data audioclip
```

This launches a jupyter instance in a container with all required dependencies already installed. You can then access it at `localhost:7799` (you can change the port number in the `docker-compose.yml` file). Password is `berlin22`


