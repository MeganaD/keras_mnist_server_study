import falcon
from digit.digit import Digit

api = application = falcon.API()


# handwriting digit recognition
digit = Digit()
api.add_route('/digit', digit)
