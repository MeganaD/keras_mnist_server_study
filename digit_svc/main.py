# coding: utf-8
from flask import Flask, request
from flask_restful import Resource, Api
import model

app = Flask(__name__)
api = Api(app)

api.add_resource(model.Digit, '/digit',
    resource_class_kwargs={ 'request': request})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False)