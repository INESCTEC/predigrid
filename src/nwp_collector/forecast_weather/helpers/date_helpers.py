from datetime import timedelta


def set_date_parameter_url(single_date):

    date_param = {'time_start': single_date.strftime('%Y-%m-%dT00:00:00Z'),
                  'time_end': (single_date + timedelta(days=4)).strftime('%Y-%m-%dT00:00:00Z')}

    return date_param
