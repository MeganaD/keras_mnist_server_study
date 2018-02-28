# mnist based digit predic server study
using keras, tensorflow, flask, gunicorn

---
### run server :   
### ```gunicorn digit_api:app -b 127.0.0.1:5000 -w 1```
server test : ``` python digit_api.py ```  

---
test server : ``` python digit.py ```  
test : ``` curl -X POST -F image=@587.png 'http://localhost:5000/digit'```  
test page : ```http://localhost:5000```

