from flask import Flask, jsonify, request
from tensorflow import keras
import numpy as np
from flask_cors import CORS

import image

import socket

app = Flask(__name__)

model = keras.models.load_model('model.h5')

# Cross Origin Resource Sharing (CORS) handling. CORS - механизм, который позволяет веб-страницам получать доступ к
# ресурсам (например, изображениям, скриптам) на других доменах. Это важно, когда веб-страница загружается
# с одного домена, а ресурсы, необходимые для ее отображения, находятся на другом домене.
CORS(app)


@app.route('/image', methods=['POST'])
def image_post_request():
    x = image.convert(request.json['image'])
    y = model.predict(x.reshape((1, 28, 28, 1))).reshape((10,))
    n = int(np.argmax(y, axis=0))
    y = [float(i) for i in y]
    ip = socket.gethostbyname_ex(socket.getfqdn())[2][0]
    return jsonify({'result': y, 'digit': n, 'ip': ip})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
