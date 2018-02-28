import falcon
from .digit import Digit

api = application = falcon.API()

digit = Digit()
api.add_route('/digit', digit)


# 샘플테스트 페이지 표시용
# api.add_static_route('/test', '/templates/test.html') # <- 절대경로만 사용할수있다...별도 웹서버를 사용해야한다.
class testpage(object):
    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.content_type = 'appropriate/content-type'
        # gunicorn을 실행하는 경로 기준으로 잡아야함
        with open('./templates/test.html', 'r') as f: 
            resp.content_type = 'text/html'
            resp.body = f.read()
            resp.status = falcon.HTTP_200
api.add_route('/', testpage())