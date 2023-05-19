FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
MAINTAINER LMC4S
LABEL description="spline-gan"
EXPOSE 6006

RUN  apt update && apt -y install git \
	&& git clone https://github.com/LMC4S/robust-spline-GAN-pytorch spline-gan \
	&& pip install session_info==1.0.0 scikit-learn==1.2.1 pandas==1.5.2 scipy==1.10.1
	
COPY . /home/workspace/spline-gan
WORKDIR /home/workspace/spline-gan
RUN chmod +x ./examples/setting1/run.sh
CMD ["/bin/bash",  "-c",  "./examples/setting1/run.sh"]



