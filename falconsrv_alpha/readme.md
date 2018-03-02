# MNIST based Digit Recognition Server
using keras, tensorflow, falcon, gunicorn

---
### run server :   
```gunicorn digit.app -b 0.0.0.0:5000 -w 3```
  
---
test page :  
```http://localhost:5000 ```
  
image file test :  
 ``` curl -X POST -F image=@587.png 'http://localhost:5000/digit'```  
