# For christmas, all weekdays can possibly have a bridgeday before/after # noqa
# for other days, only 2- Tuesday and 4- Thursday

ALL_WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday"]

RELEVANT_HOLI2BRIDGEDAYS = {
    'portugal': {"Christmas": ALL_WEEKDAYS,
                 "other": ["Tuesday", "Thursday"]},
    'france': {"Noël": ALL_WEEKDAYS,
               "other": ["Tuesday", "Thursday"]}
}


# For christmas, all the weekdays are valid bridgedays,
# for other days, only 1- Monday and 5-Friday
RELEVANT_BRIDGEDAYS = {
    "portugal": {"Christmas": ["Monday", "Tuesday", "Thursday", "Friday"],
                 "other": ["Monday", "Friday"]},
    "france": {"Noël": ["Monday", "Tuesday", "Thursday", "Friday"],
               "other": ["Monday", "Friday"]}
}


# Important bridges (for other bridges there are no post-processing):
RELEVANT_HOLIBRIDGES = {
    'portugal': ["Christmas",
                 "New Year",
                 "Assunção de Nossa Senhora",
                 "Sexta-feira Santa",
                 "Dia da Liberdade",
                 "Dia de Todos os Santos"],
    'france': ["Noël",
               "Jour de l'an",
               "Lundi de Pâques",
               "Fête du Travail",
               "Armistice 1945",
               "Ascension",
               "Fête nationale",
               "Assomption",
               "Armistice 1918",
               ],
}

# Important holidays (conditional filter in self.holidays_fix)
IMPORTANT_HOLIDAYS = {
    "portugal": ["Christmas", "New Year", "Easter"],
    "france": ["Noël", "Jour de l'an",
               'Lundi de Pâques']
}
