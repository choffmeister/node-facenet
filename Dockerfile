FROM ubuntu:17.10
LABEL maintainer="Huan LI <zixia@zixia.net>"

ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL          C.UTF-8

RUN apt-get update && apt-get install -y \
      build-essential \
      curl \
      g++ \
      git \
      graphicsmagick \
      iputils-ping \
      libcairo2-dev \
      libjpeg8-dev \
      libpango1.0-dev \
      libgif-dev \
      python2.7 \
      python3.6 \
      python3.6-dev \
      python3-venv \
      sudo \
      tzdata \
      vim \
  && rm -rf /var/lib/apt/lists/*

RUN curl -sL https://deb.nodesource.com/setup_8.x | bash - \
  && apt-get update && apt-get install -y nodejs \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir /facenet /workdir

# Add facenet user.
RUN groupadd -r facenet && useradd -r -m -G audio,video,sudo -g facenet -d /facenet facenet \
  && chown -R facenet:facenet /facenet /workdir \
  && echo "facenet ALL=NOPASSWD:ALL" >> /etc/sudoers
USER facenet

WORKDIR /facenet
COPY . .
RUN sudo chown -R facenet /facenet \
  && npm install \
  && npm run dist \
  && sudo ln -s /usr/lib/node_modules /node_modules \
  && sudo ln -s /facenet/node_modules/* /node_modules/ \
  && sudo ln -s /facenet /node_modules/facenet \
  && sudo rm -fr /tmp/* ~/.npm

EXPOSE 8080

VOLUME [ "/workdir" ]
CMD [ "node", "dist/examples/server.js" ]
