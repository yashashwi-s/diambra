FROM ghcr.io/diambra/arena-base-on3.10-bullseye:main

RUN apt-get -qy update && \
    apt-get -qy install libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
ENTRYPOINT [ "python", "/app/agent.py" ]
