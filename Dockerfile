# Use Ubuntu's current LTS
FROM python:3.11.8-slim-bookworm

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
# Make sure to not install recommends and to clean the 
# install to minimize the size of the container as much as possible.
# RUN apt-get update && \
#     apt-get install --no-install-recommends -y python3 && \
#     apt-get install --no-install-recommends -y python3-pip && \
#     apt-get install --no-install-recommends -y python3-venv
# RUN cd /usr/src \     
#     && wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz \     
#     && tar -xzf Python-3.11.0.tgz \     
#     && cd Python-3.11.0 \     
#     && ./configure --enable-optimizations \     
#     && make altinstall 
#     # apt-get clean
#     # rm -rf /var/lib/apt/lists/*

# Set the working directory within the container
WORKDIR /app

# Copy necessary files to the container
COPY requirements.txt .
COPY __init__.py .
COPY main.py .
COPY download_models.py .
COPY dataset.py .
COPY load_reader.py .
COPY ranker.py .
COPY prompt.py .
# Create a virtual environment in the container
RUN python3 -m venv .venv
# Activate the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
    # Install Python dependencies from the requirements file
RUN pip install --upgrade pip wheel --default-timeout=100 && \
   #  pip install git+https://github.com/spotify/annoy.git@main && \
    pip install -r requirements.txt --no-cache-dir && \    
    # Get the models from Hugging Face to bake into the container
    python download_models.py && \
    python dataset.py && \
    python load_reader.py && \
    python ranker.py && \
    python prompt.py

# Make port 6000 available to the world outside this container
EXPOSE 6000

ENTRYPOINT [ "python3" ]

# Run main.py when the container launches
CMD [ "main.py" ]
