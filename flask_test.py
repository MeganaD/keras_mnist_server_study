# -*- coding: utf-8 -*-

import flask
from flask import render_template
import numpy as np
import cv2 as cv
from keras.models import load_model
from keras import backend as K

# import io
# import keras
# from keras.models import load_model, model_from_yaml
# from keras.preprocessing.image import img_to_array

app = flask.Flask(__name__)
model = None

def model_load():
    global model 
    model = load_model('model.h5')

    # # 분리하여 로드하는 경우... 컴파일이 필요한듯.
    # yaml_file = open('model.yaml', 'r')
    # loaded_model_yaml = yaml_file.read()
    # yaml_file.close()
    # model = model_from_yaml(loaded_model_yaml)
    # model.load_weights("model_weight.h5")
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #           optimizer=keras.optimizers.Adadelta(),
    #           metrics=['accuracy'])

    model._make_predict_function()
    print('-- Model Load Success. --')
    
def prepare_image(img):
    # img = np.fromstring(flask.request.files['image'].read(), np.uint8)
    # img = cv.imdecode(img,cv.IMREAD_COLOR)
    im = cv.imdecode(np.fromstring(img, np.uint8), cv.IMREAD_COLOR)
    # # 이미지 검토용
    # import matplotlib.pyplot as plt
    # plt.imshow(im)
    # plt.savefig('t_pre_0_org.png')

    # 판정이미지 사이즈
    img_rows, img_cols = 28, 28
    # 대상이미지의 높이 조정
    res_h = 100
    # 이후 글자의 최소 높이와 너비를 정해서 roi 를 설정해보자
    
    im_h,im_w = im.shape[:2]
    
    r = res_h / float(im_h)
    # https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html 참고하여 보간법 설정
    im_res = cv.resize(im,(int(im_w*r),int(im_h*r)), cv.INTER_AREA if r<1 else cv.INTER_LINEAR )
    # # 이미지 검토용
    # import matplotlib.pyplot as plt
    # plt.imshow(im_res)
    # plt.savefig('t_pre_1_res.png')

    gray = cv.cvtColor(im_res, cv.COLOR_BGR2GRAY)
    # # 이미지 검토용
    # import matplotlib.pyplot as plt
    # plt.imshow(gray)
    # plt.savefig('t_pre_2_gray.png')
    
   
    # 블러 - 수직으로 좀더
    im_proc = cv.GaussianBlur(gray, (11, 17), 0)
    
    # 윤곽을 따기위해 반전, 하얀 배경에 쓴 글자를 인식하는 문제이므로 단순 2진화 사용
    thresh = cv.threshold(im_proc, 120, 255, cv.THRESH_BINARY_INV)[1]
    # # 이미지 검토용
    # import matplotlib.pyplot as plt
    # plt.imshow(thresh)
    # plt.savefig('t_pre_4_thresh.png')

    # 윤곽 추출, 최외곽, 단순
    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]

    # 문자영역 분할
    rects = []
    im_w = im_res.shape[1]
    for i, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)
        
        # 면적이 작으면 윤곽에서 제외
        if w * h < 50: continue 
        
        # 면적의 70%이상을 차지하면 윤곽에서 제외
        if w * h > im_w * res_h * 0.7: continue 
        
        rects.append((x, y, w, h))
        
        #현재는 1줄로 입력받으므로 여러줄인경우는 검토하지 않음
        #여러줄인경우 줄에따른 인덱스도 필요함
        #y2 = round(y / 10) * 10 # Y좌표 맞추기
        #index = y2 * im_w  + x
        #rects.append((index, x, y, w, h))

    # x를 기준으로 정렬
    rects = sorted(rects, key=lambda x:x[0]) 
    # print('refined rects:',len(rects))
   
    X = []
    # im_data = []

    # 분할영역 이미지 추출
    for i, r in enumerate(rects):
        x, y, w, h = r
        
        # rect 부분 이미지 추출
        num = gray[y:y+h, x:x+w] 
        
        # 반전 : mnist를 사용하므로 이에 맞게 반전
        num = 255 - num 
        
        # 여백이 있는 정사각형 중앙으로 대상 옮기기
        ww = round((w if w > h else h) * 1.6) 
        spc = np.zeros((ww, ww))
        wy = (ww-h)//2
        wx = (ww-w)//2
        spc[wy:wy+h, wx:wx+w] = num
        
        #mnist 용으로 리사이즈
        num = cv.resize(spc, (img_rows, img_cols))
        
        # 판정 검토용 이미지 보관
        # cv.imwrite(str(i)+"-num.PNG", num) 
        # im_data.append(num)    
        
        # 데이터 정규화
        num = num.reshape(img_rows * img_cols)
        num = num.astype("float32") / 255
        X.append(num)


    # CNN으로 처리하기위한 reshape
    npX=np.array(X)

    # 체널 타입 설정
    if K.image_data_format() == 'channels_first':
        npX = npX.reshape(npX.shape[0], 1, img_rows, img_cols)
    else:
        npX = npX.reshape(npX.shape[0], img_rows, img_cols, 1)

    # print('* Prepare image Success.')
    return npX

# 테스트페이지 표시용
@app.route('/')
def test_page():
    return render_template('test.html')

@app.route("/digit", methods=["POST"])
def get_digit():
    data = {"success": False}

    if model is None:
        model_load()

    if flask.request.method == "POST" :
        img = None

        # 파일 업로드시 처리
        if flask.request.files.get("image"):
            img = flask.request.files['image'].read()
        
        # ajax post 처리
        elif 'image' in flask.request.form:
            req_img = flask.request.form.get("image")
            img = re.sub('^data:image/.+;base64,', '', req_img)
            img = base64.b64decode(img)

        
        if img is not None:
            # # 검토용 파일저장
            # with open('t_req_org.png','wb') as output:
            # output.write(img)
            npX = prepare_image(img)
            results = model.predict_classes(np.array(npX))
            data["digits"] = results.tolist()
            data["number"] = int(''.join(str(n) for n in results ))
            data["success"] = True

    print(data)
    return flask.jsonify(data)

if __name__ == '__main__':
    app.debug = True
    app.run()

