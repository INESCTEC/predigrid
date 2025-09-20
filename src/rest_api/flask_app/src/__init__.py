
__title__ = 'Forecasting Services REST API'
__version__ = '1.0.0'
__author__ = 'Ricardo Andrade / jrsa@inesctec.pt'


# module level doc-string
__doc__ = """
REST API for Predigrid

API methods:
    * '/api/forecast/getForecast/' - Retrieve forecast data from forecasting services database.
    * '/api/forecast/getMeasurements/' - Retrieve measurements data from forecasting services database.
    * '/api/forecast/measurements/' - Send measurements data to forecasting services database.
    * '/api/user/register/' - Register user in forecasting services database (requires admin privileges).
    * '/api/user/list/' - List users in forecasting services database (requires admin privileges).
    * '/api/user/permissions/' - Update user permissions/privileges (requires admin privileges).
    * '/api/user/login/' - User Login into the REST API.
    * '/api/user/password_reset/' - Allows user to reset old password.
    * '/api/forecast/testService/' - Check if api service is working.
    
"""
