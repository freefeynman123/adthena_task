FROM python:3.7

WORKDIR /app

# Install poetry:
RUN pip install poetry

# Copy files required to install packages.
COPY poetry.lock pyproject.toml /app/
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction
# Copy in everything else and install:
COPY . .
RUN poetry install --no-interaction
RUN python src/adthena_task/training/main.py
