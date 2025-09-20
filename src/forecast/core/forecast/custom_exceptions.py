import sys
import traceback
import cassandra.cluster

from functools import wraps
from cassandra.connection import ConnectionException


class CustomExceptions(BaseException):
    pass


class FeatureEngException(CustomExceptions):
    pass


class ModelTrainException(CustomExceptions):
    pass


class ModelFitException(CustomExceptions):
    pass


class ModelLoadException(CustomExceptions):
    pass


class MissingInputsException(CustomExceptions):
    pass


class ModelUpdateException(CustomExceptions):
    pass


class ModelForecastException(CustomExceptions):
    pass


class TrainEmptyDatasetError(CustomExceptions):
    pass


class UpdateEmptyDatasetError(CustomExceptions):
    pass


class TrainScalerError(CustomExceptions):
    pass


class UpdateScalerError(CustomExceptions):
    pass


class LoadScalerError(CustomExceptions):
    pass


class ForecastError(CustomExceptions):
    pass


class ForecastAuxError(CustomExceptions):
    pass


class ModelClassNonExistentMethod(CustomExceptions):
    pass


DATABASE_EXCEPTIONS = (cassandra.ReadFailure, cassandra.ReadTimeout,
                       cassandra.OperationTimedOut, cassandra.Timeout,
                       cassandra.FunctionFailure, cassandra.Timeout,
                       cassandra.WriteFailure, cassandra.WriteTimeout,
                       cassandra.cluster.NoHostAvailable,
                       ConnectionException)


def simple_fail_proof(exc=None, msg=None):
    """
    Decorator that will allow to handle exceptions while avoiding
    having to fill code with try statements. This will try to run function,
    retrying call in case of database error, and produces log messages in case
    of failure.

    :param exc:
    :param msg:
    :return:
    """
    def decorator_fail_proof(func):
        @wraps(func)
        def wrapper_fail_proof(self, *args, **kwargs):
            # Counter for retries in case of database failure
            tries = 3
            while tries:
                try:
                    # Run decorated function
                    return func(self, *args, **kwargs)
                except DATABASE_EXCEPTIONS as ex:
                    # Database exceptions will either trigger a retry or
                    # will raise exception after maximum number of retries
                    tries -= 1
                    if tries > 0:
                        self.logger.warning(f"Retrying connection to database. [{-(tries-3)}/3]. Cause: {repr(ex)}") # noqa
                    if tries:
                        continue
                    else:
                        self.logger.error(f"Unable to connect to database. Cause: {repr(ex)}") # noqa
                        raise ex
                except BaseException as ex:
                    # Other exceptions will produce log messages
                    # and break while statement
                    self.logger.error(msg or f"{func.__name__} ({repr(ex)})") # noqa
                    if self.unit_tests_mode:
                        raise ex
                    break
                    # raise exc(ex)
        return wrapper_fail_proof
    return decorator_fail_proof
