from flask_restful import Resource
#------ Dashboard -------#
class Dashboard(Resource):
    def get(self):
        return {'url':'in dashboard'}