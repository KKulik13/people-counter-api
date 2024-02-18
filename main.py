from flask import Flask, request, render_template
from flask_restful import Resource, Api
import cv2
import requests
import numpy as np

app = Flask(__name__)
api = Api(app)

# img = cv2.imread('image/dworzec.jpeg')     # funkcja zwraca odczytane zdjęcie
# cv2.imshow('image', img)                   # wyświetlenie zdjęcia
# cv2.waitKey(0)                             # czeka na wciśnięcie klawisza
# cv2.destroyAllWindows()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


@app.route('/index', methods=['GET', 'POST'])
def PeopleCounterForm():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        if 'link' in request.form:
            link = requests.get(request.form['link'], stream=True).raw
            arr = np.asarray(bytearray(link.read()), dtype="uint8")
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        else:
            boxes, weights = None

        return render_template('index.html', link=len(boxes))


class PeopleCounterUrl(Resource):
    def get(self):
        url = request.args.get('url')
        url = requests.get(url, stream=True).raw
        arr = np.asarray(bytearray(url.read()), dtype="uint8")
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        return {'count': len(boxes)}


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('images/dworzec.jpeg')
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        # boxes - na jakich współrzędnych znajdują  się osoby
        # print(type(img))
        # print(type(boxes))  # lista
        # print(img.shape)
        return {'Ilość osób na zdjęciu': len(boxes)}


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class HelloWorld2(Resource):
    def get(self):
        return {'hello': 'world2'}


api.add_resource(HelloWorld, '/test')
api.add_resource(HelloWorld2, '/test2')
api.add_resource(PeopleCounter, '/')
api.add_resource(PeopleCounterUrl, '/url')


if __name__ == '__main__':
    app.run(debug=True, port=8080)
