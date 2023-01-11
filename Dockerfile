FROM python:3.9

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app"]