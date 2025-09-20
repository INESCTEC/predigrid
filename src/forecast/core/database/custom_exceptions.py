
class CustomExceptions(BaseException):
    pass


class DatabaseEngineException(CustomExceptions):
    pass


class QueryDataException(CustomExceptions):
    pass


class DatabaseInsertException(CustomExceptions):
    pass
