FROM python:3.8

RUN python -m pip install flask flask-cors gunicorn numpy tensorflow pillow==9.5.0

WORKDIR /app

ADD image.py image.py
ADD mnist_recognizer.py mnist_recognizer.py
ADD model.h5 model.h5

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "mnist_recognizer:app" ]

