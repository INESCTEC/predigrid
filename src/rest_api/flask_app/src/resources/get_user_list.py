
from flask import request
from flask_restful import Resource

from ..common.UserClass import UserClass
from ..common.databases import CassandraDB
from ..resources.post_login import token_required_admin


class GetUserList(Resource, UserClass):
    method_decorators = [token_required_admin]

    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger')
        self.config = kwargs.get('config')

    def get(self):
        """
                Retrieve the list of registered users
                ---
                tags:
                  - Users
                security:
                  - bearerAuth: []   # requires admin token

                parameters:
                  - name: Authorization
                    in: header
                    type: string
                    required: true
                    description: >
                        Bearer JWT token with `admin` privileges.
                        Example: `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
                responses:
                  200:
                    description: List of users successfully retrieved
                    content:
                      application/json:
                        example:
                          - username: "admin"
                            email: "admin@example.com"
                            role: "admin"
                            registered_at: "2025-01-10 14:22:31"
                            last_login: "2025-09-18 09:42:11"
                          - username: "jane"
                            email: "jane@example.com"
                            role: "user"
                            registered_at: "2025-03-04 08:15:00"
                            last_login: "2025-09-17 17:20:45"
                  401:
                    description: Unauthorized (invalid or missing admin token)
                    content:
                      application/json:
                        example:
                          code: 0
                          message: "Invalid token."
                  500:
                    description: Internal server error
                    content:
                      application/json:
                        example:
                          code: 0
                          message: "Internal Server Error."
                """
        # Connect to Cassandra DB:
        user_table = self.config.TABLES["users"]
        db_con = CassandraDB.get_cassandra_instance(config=self.config)
        user_data = db_con.read_query("select * from {};".format(user_table))
        user_data.loc[:, 'registered_at'] = user_data["registered_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
        user_data.loc[:, 'last_login'] = user_data["last_login"].dt.strftime("%Y-%m-%d %H:%M:%S")
        user_data_json = user_data.to_dict(orient='records')
        return user_data_json, 200
