import numpy as np

from core.database.helpers.cql_helpers import (
    save_obj_to_cql,
    load_obj_from_cql
)


def verify_data_statistics(con, network_id, pt_id, model_id, data_stats):
    old_stats = load_obj_from_cql(con, network_id, pt_id, "stats", model_id)

    if old_stats is None:
        save_obj_to_cql(con, network_id, pt_id, "stats", model_id, data_stats)
        return True

    consistency_change = []
    for k, v in data_stats.items():
        try:
            prev_stats = old_stats.get(k)
            _prevmax = prev_stats.get("max", np.nan)
            _prevmin = prev_stats.get("min", np.nan)
            _prevcount = prev_stats.get("count", np.nan)
            new_stats = data_stats.get(k)
            _newmax = new_stats.get("max", np.nan)
            _newmin = new_stats.get("min", np.nan)
            _newcount = new_stats.get("count", np.nan)

            # if any element of comparison (_newmax, _prevmax, etc.)
            # is nan, the condition is false always. Tries to update
            # but crashes later on train phase with exception. so it's ok.
            if (_newmax > _prevmax) or (_newmin < _prevmin) or (abs(_newcount - _prevcount) > 744): # noqa
                if _newmax > _prevmax:
                    old_stats[k]["max"] = _newmax
                    # print(f"NEW {k} is bigger! {_newmax} > {_prevmax}")
                if _newmin < _prevmin:
                    old_stats[k]["min"] = _newmin
                    # print(f"NEW {k} is smaller! {_newmin} > {_prevmin}")
                if abs(_newcount - _prevcount) > 48:
                    old_stats[k]["count"] = _newcount
                    # print("Increase in historical values {k} ! "
                    #       "{_newmin} < {_prevmin}")
                consistency_change.append(False)
            else:
                consistency_change.append(True)
        except AttributeError:
            consistency_change.append(False)

    # if one element of to_update is False, updatable is False, else it is True
    consistency_change = not all(consistency_change)
    if consistency_change:
        # overwrites previous file with new stats
        save_obj_to_cql(con, network_id, pt_id, "stats", model_id, old_stats)

    return consistency_change
