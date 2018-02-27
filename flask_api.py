# -*- coding: utf-8 -*-

from flask import Flask, request,jsonify
from flask_restful import Resource, Api

import numpy as np
import cv2 as cv
from keras.models import load_model
from keras import backend as K

app = Flask(__name__)
api = Api(app)

model = None

def model_load():
    global model 
    model = load_model('model.h5')
    model._make_predict_function()
    print('-- Model Load Success. --')

class Digit(Resource):
    def __init__(self):
        pass

    def prepare_image(self, img):
        im = cv.imdecode(np.fromstring(img, np.uint8), cv.IMREAD_COLOR)
        img_rows, img_cols = 28, 28
        res_h = 100
        im_h,im_w = im.shape[:2]
        
        r = res_h / float(im_h)
        im_res = cv.resize(im,(int(im_w*r),int(im_h*r)), cv.INTER_AREA if r<1 else cv.INTER_LINEAR )
        gray = cv.cvtColor(im_res, cv.COLOR_BGR2GRAY)

        im_proc = cv.GaussianBlur(gray, (3, 5), 0)
        thresh = cv.threshold(im_proc, 120, 255, cv.THRESH_BINARY_INV)[1]
        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]

        rects = []
        im_w = im_res.shape[1]
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if w * h < 10: continue 
            if w * h > im_w * res_h * 0.7: continue 
            rects.append((x, y, w, h))
        rects = sorted(rects, key=lambda x:x[0]) 
    
        X = []
        for i, r in enumerate(rects):
            x, y, w, h = r
            num = gray[y:y+h, x:x+w] 
            num = 255 - num 
            ww = round((w if w > h else h) * 1.6) 
            spc = np.zeros((ww, ww))
            wy = (ww-h)//2
            wx = (ww-w)//2
            spc[wy:wy+h, wx:wx+w] = num
            num = cv.resize(spc, (img_rows, img_cols))
            num = num.reshape(img_rows * img_cols)
            num = num.astype("float32") / 255
            X.append(num)

        npX=np.array(X)

        if K.image_data_format() == 'channels_first':
            npX = npX.reshape(npX.shape[0], 1, img_rows, img_cols)
        else:
            npX = npX.reshape(npX.shape[0], img_rows, img_cols, 1)

        # print('* Prepare image Success.')
        return npX

    def post(self):
        data = {"success": False}

        if request.files.get("image"):
            img = request.files['image'].read()
            npX = self.prepare_image(img)
            results = model.predict_classes(np.array(npX))
            data["digits"] = results.tolist()
            data["number"] = int(''.join(str(n) for n in results ))
            data["success"] = True

        return jsonify(data)

api.add_resource(Digit, '/digit')

if __name__ == '__main__':
    model_load()
    app.run(debug=False)