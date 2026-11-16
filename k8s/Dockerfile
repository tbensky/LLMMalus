# The lab harness. Standard library only, plus matplotlib so plots are PNGs.
#
#   ./malusctl build        (or the docker buildx command in k8s/README.md)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /usr/src/app

# Dependencies first, so a code change does not reinstall them.
COPY k8s/requirements.txt /usr/src/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

# The git commit the image was built from; each run summary records it.
ARG BUILD_VERSION=unknown
ENV MALUS_VERSION=${BUILD_VERSION}

CMD ["sh", "k8s/entrypoint.sh"]
