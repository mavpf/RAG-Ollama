FROM ollama/ollama

#Install NVIDIA Container toolkit, for GPU USAGE

RUN apt-get update \
    && apt-get install -y curl gpg
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    |  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
RUN curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
RUN sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
RUN apt-get update
RUN apt-get install -y nvidia-container-toolkit

#Expose port 11434

EXPOSE 11434