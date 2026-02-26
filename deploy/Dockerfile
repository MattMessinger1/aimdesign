FROM python:3.11-slim

WORKDIR /app

# System deps (tiny set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
 && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8866

ENTRYPOINT ["tini", "--"]
CMD ["voila", "--no-browser", "--port=8866", "--Voila.ip=0.0.0.0", "tools/obf_no_lab_simulation.ipynb"]
