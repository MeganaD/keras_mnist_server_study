import numpy as np
import cv2 as cv

def prepare_image(img):
    # img = np.fromstring(flask.request.files['image'].read(), np.uint8)
    # img = cv.imdecode(img,cv.IMREAD_COLOR)
    im = cv.imdecode(np.fromstring(img, np.uint8), cv.IMREAD_COLOR)

    # 판정이미지 사이즈
    img_rows, img_cols = 28, 28
    # 대상이미지의 높이 30으로 조정
    res_h = 30
    # 이후 글자의 최소 높이와 너비를 정해서 roi 를 설정해보자
    
    im_h,im_w = im.shape[:2]
    
    r = res_h / float(im_h)
    # https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html 참고하여 보간법 설정
    im_res = cv.resize(im,(int(im_w*r),int(im_h*r)), cv.INTER_AREA if r<1 else cv.INTER_LINEAR )
    gray = cv.cvtColor(im_res, cv.COLOR_BGR2GRAY)
    
    # 블러 - 수직으로 좀더
    im_proc = cv.GaussianBlur(gray, (3, 5), 0)
    
    # 윤곽을 따기위해 반전, 하얀 배경에 쓴 글자를 인식하는 문제이므로 단순 2진화 사용
    thresh = cv.threshold(im_proc, 120, 255, cv.THRESH_BINARY_INV)[1]

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

    # 백엔드 처리이나 텐서플로로 고정되어있으니 생략
    # from keras import backend as K
    # if K.image_data_format() == 'channels_first':
    #     npX = npX.reshape(npX.shape[0], 1, img_rows, img_cols)
    # else:
    #     npX = npX.reshape(npX.shape[0], img_rows, img_cols, 1)

    npX = npX.reshape(npX.shape[0], img_rows, img_cols, 1)

    print('* Prepare image Success.')
    return npX
