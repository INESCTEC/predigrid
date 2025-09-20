import jwt
import datetime as dt

from ..common.UserClass import UserClass


def generate_auth_token(username, permissions, password, secret_key):
    token = jwt.encode(
        {
            'user': username,
            'permissions': permissions,
            'password': password
        },
        secret_key
    )

    print("token:", token)
    return token


def register_admin_user(db_con, config_class, username, password, logger):
    permissions = "admin"
    users_table = config_class.TABLES["users"]
    uc = UserClass()
    pwd_hash = uc.hash_password(password=password)
    userinfo = "added by createsuperuser.py method"
    last_login = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    register_date = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    q = (f"INSERT INTO {users_table} ("
         f"username, "
         f"last_login, "
         f"password_hash, "
         f"permissions, "
         f"userinfo, "
         f"registered_at) "
         f"VALUES ("
         f"'{username}', "
         f"'{last_login}', "
         f"'{pwd_hash}', "
         f"'{permissions}', "
         f"'{userinfo}', "
         f"'{register_date}')")
    try:
        db_con.execute_query(q)
        logger.info("User {} registered with success.".format(username))
    except BaseException as ex:
        msg = "Failed to register user {}. ERROR: {}".format(username,
                                                             repr(ex))
        logger.exception(msg)
