# mnist based digit predic server study

using keras, tensorflow
flask, falcon studying

test :   
``` curl -X POST -F image=@587.png 'http://localhost:5000/digit'```
  
test용 web page :
 ```/templates/test.html```

---
### [flask_test.py](flask_test.py)
플라스크 단순 예제  
모듈 분리 및 마이크로 서비스 형태로 개선 필요  
server : ``` python flask_test.py ```  
  
  
### [flask_api.py](flask_api.py)
Flask-RESTful API study  
rest 서비스 형태 구현  
뷰함수에서 클래스로 변경  
server : ``` python flask_api.py ```  
  
  
## [digit_svc](digit_svc)
파일 분리, gunicorn 테스트  
server : ```/digit_svc/gunicorn main:app -b 127.0.0.1:5000```  
제대로 한건지 모르겠음.  T^T  
파일을 분리하면서 req 전체를 던져줘야 함  <--뭔가 잘못 한듯
import가 완전히 분리되지 않음  
  
  
## [digit_fal](digit_fal)
falcorn framework 테스트  
server : ```gunicorn --reload digit_fal.app -b 127.0.0.1:5000```  
플라스크 레스트 보다 훨씬 깔끔하게 구현된 듯함. 
자체 서버가 없으므로 테스트도 gunicorn등을 사용해야함
사용 해 보니 플라스크 공부가 부족했다고 생각됨
그래도 이게 일단 완성도가 있어보이니 이쪽으로 사용
