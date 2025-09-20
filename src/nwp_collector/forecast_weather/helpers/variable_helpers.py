def meteogalicia_variables():
    return ['ulev1', 'ulev2', 'ulev3', 'visibility', 'vlev1', 'vlev2', 'vlev3', 'weasd', 'wind_gust',
            'cape', 'cfh', 'cfl', 'cfm', 'cft', 'cin','meteograms', 'mod', 'mslp', 'conv_prec', 'dir', 'land_use',
            'lhflx', 'lwflx', 'lwm', 'pbl_height', 'prec', 'rh', 'shflx', 'snow_prec', 'snowlevel', 'sst', 'swflx',
            'temp', 'topo', 'u', 'v', 'HGT500', 'HGT850', 'HGTlev1', 'HGTlev2', 'HGTlev3', 'T500', 'T850']


def solar_variables():
    return ['cft', 'cfh', 'cfl', 'cfm', 'swflx', 'temp']


def wind_variables():
    return ['dir', 'mod', 'u', 'ulev1', 'ulev2', 'ulev3',
            'v', 'vlev1', 'vlev2', 'vlev3']