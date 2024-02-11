from flask import Flask
from flask_restful import Resource, Api
import cv2


app = Flask(__name__)
api = Api(app)

# img = cv2.imread('image/dworzec.jpeg')     # funkcja zwraca odczytane zdjęcie
# cv2.imshow('image', img)                   # wyświetlenie zdjęcia
# cv2.waitKey(0)                             # czeka na wciśnięcie klawisza
# cv2.destroyAllWindows()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('images/dworzec.jpeg')
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        boxes, weights = hog.detectMultiScale(img, winStride=(8, 8))
        # boxes - na jakich współrzędnych znajdują  się osoby
        # print(type(img))
        # print(type(boxes))  # lista
        # print(img.shape)
        return {'count test': len(boxes)}


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class HelloWorld2(Resource):
    def get(self):
        return {'hello': 'world2'}


api.add_resource(HelloWorld, '/test')
api.add_resource(HelloWorld2, '/test2')
api.add_resource(PeopleCounter, '/')


if __name__ == '__main__':
    app.run(debug=True, port=8080)
