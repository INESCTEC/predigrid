![logo-predigrid.png](./docs/img/logo-predigrid.png)

# PREDIGRID Forecasting service (Docker Containers Stack Deployment -- README)

This repository provides a modular forecasting system for load and generation 
at both low-voltage (LV) and medium-voltage (MV) levels, developed by INESC TEC. 
It supports forecasting of active and reactive power and is composed of several
modules that handle the different tasks required in a forecasting workflow, 
enabling flexible and accurate use across grid applications.
The developed modules are:

    1. Data Disposer - For data processing (netload computation and outlier detection)

    2. Forecast module - For model training and forecast computing

    3. Metrics module - For KPI computation

    4. Weather module - For NWP retrieval and processing

    5. REST API - Platform for system configuration and data access

Documentation for each module can be found in folder **docs**

**Note:** The service requires an Apache Cassandra database to operate. The database
can be set up using the official Cassandra docker image, or by using a managed
Cassandra service. A cql file with name `db_schema.cql` can be found in the 
root of the repository.

## Configurations

In order to set the system, different environment variables need to be set.
In the folder **ENV** there are several files with the environment variables 
for each module, which are listed below. The variables with a comment between
parentheses need to be set by the user, while the rest of the variables should
be left as they are.
The environment variables for each module are the following:

### 1. Data Disposer

    # -- Data Disposer Configs:

    # Cassandra DB Settings:
    CASSANDRA_CONTACT_POINTS=(Cassandra database host)
    CASSANDRA_PORT=(Cassandra database port)
    CASSANDRA_KEYSPACE=(Cassandra database keyspace)
    
    # Tables:
    TBL_METADATA=installations (table in database for installations metadata)
    TBL_RAW=raw (table in database for raw measures)
    TBL_NETLOAD=net (table in database for netload data)
    TBL_PROCESSED=processed (table in database for processed)
    
    # System:
    N_JOBS = -1 (number of parallel processes for running tasks)

### 2. Forecast module

    Does not need environment variables

### 3. Metrics module

    # -- Metrics settings:
    
    # DB connection
    CASSANDRA_CONTACT_POINTS=(Cassandra database host)
    CASSANDRA_PORT=(Cassandra database port)
    CASSANDRA_KEYSPACE=(Cassandra database keyspace)
    
    # Tables
    TABLE_FORECASTS=forecast
    TABLE_REAL_DATA=net
    TABLE_CLIENTS=installations
    TABLE_METRICS_YEAR=metrics_year
    TABLE_METRICS_2WEEKS=metrics_2wk

### 4. Weather module

    # -- Weather settings:
    
    # Cassandra connection
    CASSANDRA_CONTACT_POINTS=(Cassandra database host)
    CASSANDRA_PORT=(Cassandra database port)
    CASSANDRA_KEYSPACE=(Cassandra database keyspace)
    
    # Forecast options
    SOURCE=wrf_12km
    
    # Grid limits
    WEST=(NWP Grid limit west)
    SOUTH=(NWP Grid limit south)
    EAST=(NWP Grid limit east)
    NORTH=(NWP Grid limit north)
    
    # Database Tables
    INSTALLATIONS_TABLE=installations
    GRID_TABLE=nwp_grid
    SOLAR_TABLE=nwp_solar
    WIND_TABLE=nwp_wind

### 5. REST API module

    # -- REST API Settings:

    # Base Settings:
    TOKEN_DURATION=(Token desired duration)
    SECRET_KEY=(Private key for admin access)
    EXPOSED_PORT_NGINX=5000

    # Tables:
    TBL_METADATA=installations
    TBL_MODELS=models_info
    TBL_USERS=users_rest
    TBL_RAW=raw
    TBL_NETLOAD=net
    TBL_NWP_GRID=nwp_grid
    TBL_FORECAST=forecast
    TBL_METRICS_LONGTERM=metrics_year
    TBL_METRICS_SHORTTERM=metrics_2wk
    
    # -- Cassandra DB Settings:
    CASSANDRA_CONTACT_POINTS=(Cassandra database host)
    CASSANDRA_PORT=(Cassandra database port)
    CASSANDRA_KEYSPACE=(Cassandra database keyspace)

For the remainder of the variables for this module, please do not edit.

## Deployment:

In order to set up the service, you will need the following software:

- Docker Community Edition (CE) Engine
- Docker Compose Tool
- Python 3.8
- pip

After that, in a command line with this repository as working directory, just run one of the commands:

```
docker-compose -f docker-compose.yml up -d
```

**Note:** This will the REST API (nginx + app) containers. These containers are persistent and will remain connected untill stopped by user.

On a command line, navigate to the root of the repository and install the required python packages:

```
pip install -r src/forecast/requirements.txt
```

## Initial steps

There are a few initial configuration steps to the system:


### 1. Register Superuser in REST API:

This will create a superuser in the RESTful API that will allow to perform
several tasks, such as configuring the system (registering installations, creating other users, ...),
as well as getting information from the system (like forecasts or processed netload data)


```
docker exec -it predigrid_restapi_app python createsuperuser.py --username={USERNAME} --password={PASSWORD}
```

Where USERNAME and PASSWORD correspond 
to the username and password you wish to define as a superuser (or admin)

### 2. Create user accounts in REST API:
After creating the superuser, you can create other users in the system
using the appropriate REST API endpoint 
(try opening the link http://localhost:5001/apidocs/#/ 
on your browser after deployment, replacing the port number if it was 
changed in the environment variables for the REST API).

### 3. Register Installations in REST API and send raw data:
On this step you will need to register the installations you wish to forecast for
in the REST API, as well as send the historical raw data for these installations.

Acces the REST API documentation for more information.

### 4. Get Numerical Weather Predictions grid metadata

This step forms a grid using the coordinates provided in the environment variables
for the weather module and populates the database with the corresponding metadata.

```
docker-compose run --rm weather python main_get_grid.py

```

## Scheduled Tasks

The system operates by means of several tasks to be performed programmatically,
i.e., by either executing them manually or by setting up automated tasks in
your operating system. These tasks are:


### Run Data Disposer

Power data netload computation and processing. This should be run at least once before every forecast computation.

```
docker-compose run --rm data_disposer run_disposer
```

To run data disposer netload calculation task considering a specific lookback period (in days) & last date reference:

```
docker-compose run --rm data_disposer python run_netload.py --n_days=60 --last_date=2021-08-11
```

Similarly, to run data disposer data cleansing task considering a specific lookback period (in days) & last date reference:

```
docker-compose run --rm data_disposer python run_cleanse.py --n_days=60 --last_date=2021-08-11
```


### Run Weather Retrieval

Update the database with fresh Numerical Weather Predictions data. This should be run preferably once a day.

```
docker-compose run --rm weather python main_get_weather.py
```


### Run Metrics Engine

Compute and store new key performance indicators (KPIs). This routine can be run in a desired period (e.g. daily or weekly)

```
docker-compose run --rm weather python main.py
```

### Train forecasting models

Train models to be used by the forecasting service. Can be run every week or every two weeks.
An example of how to run the training routine can be found in `src\forecast\core\tasks\train_tasks\train_example.py`
Please review and edit the parameters in this file before running.
Then, from the root of the repository, run:
```
python src\forecast\core\tasks\train_tasks\train_example.py
```

### Run Forecast computation
Compute and store new forecasts. This routine can be run in a desired period (e.g. daily or weekly)
An example of how to run the forecast computation routine can be found in `src/forecast/core/tasks/forecast_tasks/forecast_example.py`
Please review and edit the parameters in this file before running.
Then, from the root of the repository, run:
```
python src\forecast\core\tasks\forecast_tasks\forecast_example.py
```

If you have any questions or suggestions, please contact the developers at:

- **José Andrade** <jose.r.andrade@inesctec.pt>
- **João Viana** <joao.p.viana@inesctec.pt>
