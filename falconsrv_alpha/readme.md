# ML services.  

using keras, tensorflow, falcon, gunicorn  
main server : [/main/api.py](/main/api.py)
### run server :   
```gunicorn main.api -b 0.0.0.0:5000```  

---
## [Digit](/digit)  
MNIST-based Handwritten Digit Recognition Service.  
### Parameters
| parameter | type | description |
| --- | --- | --- |
| image | file or<br>string | digit image |
  
test page (get) :  
```http://abcd:5000/digit ```  
  
image file test :  
``` curl -X POST -F image=@587.png 'http://abcd:5000/digit'```  
