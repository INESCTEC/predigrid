
from flask_restful import Resource



class GetTestService(Resource):
    # method_decorators = [token_required]

    def get(self):
        return {"code": 1, "message": "ok"}, 200
