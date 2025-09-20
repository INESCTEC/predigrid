SEND_MEASUREMENTS = {
    "type": "object",
    "definitions": {
        "inspectmeasurements": {
            "type": "object",
            "properties": {
                "installation_code": {"type": "string"},
                "values": {
                    "type": "array",
                    "items": [
                        {
                            "type": "object",
                            "required": ["timestamp", "units", "value",
                                         "variable_name"]
                        }
                    ]
                }
            },
            "required": ["installation_code", "values"]
        }
    },

    "properties": {
        "measurements": {
            "type": "array",
            "items": {
                "$ref": "#/definitions/inspectmeasurements"
            }
        },
    },
    "required": ["measurements"]
}

REGISTER_INSTALLATION = {
    "type": "object",
    "properties": {
        "installation_code": {
            "type": "string",
            "description": "Installation identifier.",
        },
        "country": {
            "type": "string",
            "description": "Country name. "
                           "Valid options: ['portugal', 'france', 'spain'].",
        },
        "generation": {
            "type": "integer",
            "description": "0 - installation without self-consumption; "
                           "1 - installation with self-consumption.",
        },
        "installation_type": {
            "type": "string",
            "description": "Installation type. "
                           "Valid types: ['load', 'wind', 'solar']"
        },
        "latitude": {
            "type": "number",
            "description": "Installation site latitude reference."
        },
        "longitude": {
            "type": "number",
            "description": "Installation site longitude reference."
        },
        "source_nwp": {
            "type": "string",
            "description": "NWP source reference. "
                           "Valid options: ['meteogalicia']."
        },
        "is_active": {
            "type": "integer",
            "description": "Installation forecast status. "
                           "1 - Active;"
                           "0 - Inactive."
        },
        "net_power_types": {
            "type": "string",
            "description": "Power consumption channels available. "
                           "Valid options: 'P', 'Q', 'PQ'.",
        }
    },
    "required": ["installation_code", "country", "generation",
                 "installation_type", "latitude", "longitude", "source_nwp",
                 "net_power_types"],
}

LOGIN = {
    "type": "object",
    "properties": {
        "username": {
            "description": "Missing username.",
            "type": "string"
        },
        "password": {
            "description": "Missing password.",
            "type": "string"
        },
    },
    "required": ["username", "password"]
}

RESET_PASSWORD = {
    "type": "object",
    "properties": {
        "username": {
            "description": "REST API client username.",
            "type": "string"
        },
        "old_password": {
            "description": "REST API client old password.",
            "type": "string"
        },
        "new_password": {
            "description": "REST API client new password.",
            "type": "string"
        }

    },
    "required": ["username", "old_password", "new_password"]
}


CHANGE_INST_STATUS = {
    "type": "object",
    "properties": {
        "installation_code": {
            "type": "string",
            "description": "Installation identifier.",
        },
        "is_active": {
            "type": "integer",
            "description": "Installation forecast status. "
                           "1 - Active;"
                           "0 - Inactive."
        }
    },
    "required": ["installation_code", "is_active"]
}

