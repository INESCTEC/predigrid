README file for mock generation. Explains each row in installations.csv (row = different installation setting)

Attempt | Type  | Measurements | NWP | Description | In tests |
---     | ---   | ---          | --- | ---         | ---      |
inst11  | load  | Full historical  | Full historical + Full horizon | Ideal case where all data is available | test_data_manager;test_forecast_manager |
inst12  | load_gen  | Only last month available  | Full historical + Last 48h in horizon not available | Test for "D-7/no_weather" model for the last 2 days in horizon | test_data_manager;test_forecast_manager |
inst13  | load  | No measurements data  | Full historical + Full horizon | Test case where there are no measurements data | test_data_manager |
inst14  | load  | Removed last 48h of measurements | Full historical + Full horizon | Test models_ref construction (adaptive lags - 'D-3' and 'D-7' available) | test_data_manager |
inst15  | load  | Removed last 96h of measurements | Full historical + Full horizon | Test models_ref construction (adaptive lags - only 'D-7' available) | test_data_manager |
inst16  | load  | Full historical  | No weather | Test case where no NWP data is available | test_data_manager |
inst17  | load  | Full historical  | Full historical + Incomplete horizon (last 12h not available) | Test case for model backup-mix (has NWP but not for complete horizon) | test_data_manager |
inst21  | wind  | Full historical  | Full historical | Ideal case where all data is available | test_data_manager;test_forecast_manager |
inst22  | wind  | Full historical  | Full historical + Incomplete horizon (last 12h not available) | Test case for model backup-mix (has NWP but not for complete horizon) | test_forecast_manager |
inst23  | wind  | Only last month available  | Full historical | Test for cases with short historical | test_forecast_manager
inst31  | solar | Full historical  | Full historical | Ideal case where all data is available |  test_data_manager;test_forecast_manager
inst32  | solar | Full historical  | Full historical + Incomplete horizon (last 12h not available) | Test case for model backup-mix (has NWP but not for complete horizon) | test_forecast_manager |
inst33  | solar | Only last month available  | Full historical | Test for cases with short historical | test_forecast_manager