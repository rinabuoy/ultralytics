FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
RUN apt-get update && apt-get upgrade -y && apt install -y libgl1
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt install -y git htop tmux gcc cuda-runtime-12-8

RUN pip install scikit-image tqdm numpy Pillow pandas einops opencv-python wandb cython

RUN pip install avalanche_lib
RUN pip install albumentations
RUN pip install torchsummary
RUN pip install einops
RUN pip install dataclasses
RUN pip install --upgrade transformers
RUN pip install datasets
RUN pip install huggingface_hub
RUN pip install polars
#RUN pip install ultralytics

COPY yolo26x_line_khm.py app/
COPY ultralytics app/ultralytics
ENV PYTHONPATH="/app"
CMD ["python3", "app/yolo26x_line_khm.py" ]