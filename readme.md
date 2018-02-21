# mnist based digit predic server study

using keras, tensorflow

---
### [flask_test.py](flask_test.py)
플라스크 단순 예제  
모듈 분리 및 마이크로 서비스 형태로 개선 필요  
server : ``` python flask_test.py ```  
client test : ``` curl -X POST -F image=@587.png 'http://localhost:5000/digit'```
  
  
### [flask_api.py](flask_api.py)
Flask-RESTful API study  
rest 서비스 형태 구현  
뷰함수에서 클래스로 변경  
server : ``` python flask_api.py ```  
client test : ``` curl -X POST -F image=@587.png 'http://localhost:5000/digit'```


