FROM python:3.8

RUN python -m pip install protobuf==3.20.0
RUN python -m pip install gdown
RUN gdown https://drive.google.com/uc?id=1T3WrRsiKBBqZn-LRaBQL6ulAQM-Unh3G
RUN python -m pip install tensorflow-2.12.0-cp38-cp38-linux_x86_64.whl && rm -rf tensorflow-2.12.0-cp38-cp38-linux_x86_64.whl
RUN python -m pip install flask flask-cors gunicorn pillow keras==2.11
RUN python -m pip install numpy==1.22


WORKDIR /app

ADD image.py image.py
ADD mnist_recognizer.py mnist_recognizer.py
ADD model-cpu-fix-2nd.h5 model.h5

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "mnist_recognizer:app" ]

