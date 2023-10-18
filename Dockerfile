FROM python:3.10.12

WORKDIR /app

COPY './requirements.txt' .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install opencv-python-headless

COPY . .

CMD ["python", "app.py"]
