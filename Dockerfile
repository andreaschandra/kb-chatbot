FROM python:3.11

WORKDIR /app

COPY . /app

RUN apt update -y
RUN apt install build-essential -y
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN ~/.local/bin/uv sync

CMD ["sh", "run.sh"]