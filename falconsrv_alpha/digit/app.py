import falcon
from .digit import Digit

api = application = falcon.API()

digit = Digit()
api.add_route('/digit', digit)


# 샘플테스트 페이지 표시용
class testpage(object):
    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.content_type = 'appropriate/content-type'
        with open('./testpage.html', 'r') as f: 
            resp.content_type = 'text/html'
            resp.body = f.read()
            resp.status = falcon.HTTP_200
api.add_route('/', testpage())