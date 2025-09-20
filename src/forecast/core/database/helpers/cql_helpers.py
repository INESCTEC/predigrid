import codecs
import pickle
import datetime as dt

from itertools import accumulate, chain, repeat, tee


def chunk(xs, n):
    assert n > 0
    L = len(xs)
    s, r = divmod(L, n)
    widths = chain(repeat(s + 1, r), repeat(s, n - r))
    offsets = accumulate(chain((0,), widths))
    b, e = tee(offsets)
    next(e)
    return [xs[s] for s in map(slice, b, e)]


def save_obj_to_cql(con, inst_id, obj_id, model_id, model_obj,
                    split_serial=False, split_serial_size=10):
    last_update = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    serial_obj = codecs.encode(pickle.dumps(model_obj), "base64").decode()
    query = "INSERT INTO " \
            "objects (id, model_id, obj_id, obj_serial, last_update) " \
            "VALUES " \
            "('{inst_id}', '{model_id}', '{obj_id}', '{obj}', '{last_update}')"
    if not split_serial:
        q = query.format(obj_id=obj_id, model_id=model_id,
                         inst_id=inst_id,
                         obj=serial_obj, last_update=last_update)
        con.session.execute(q)
    else:
        # pieces = textwrap.wrap(serial_obj, 40000)
        pieces = chunk(serial_obj, split_serial_size)
        # import sys
        # print("Chunk size:", sys.getsizeof(pieces[0])/1000, "MB")
        for p, sp in enumerate(pieces):
            m_id = str(model_id) + '_' + str(p)
            q = query.format(obj_id=obj_id, model_id=m_id,
                             inst_id=inst_id,
                             obj=sp, last_update=last_update)
            con.session.execute(q)


def load_obj_from_cql(con, inst_id, obj_id, model_id,
                      split_serial=False, split_serial_size=10):
    query = "select obj_serial from objects " \
            "where id='{inst_id}' " \
            "and model_id='{model_id}' " \
            "and obj_id='{obj_id}';"

    serialized_obj = ''
    if not split_serial:
        q = query.format(obj_id=obj_id, model_id=model_id, inst_id=inst_id)
        q_res = con.session.execute(q)
        serialized_obj = q_res.current_rows[0]["obj_serial"]
        serialized_obj = serialized_obj[0] if len(serialized_obj) > 0 else ''
    else:
        for p in range(0, split_serial_size):
            m_id = str(model_id) + '_' + str(p)
            q = query.format(obj_id=obj_id, model_id=m_id, inst_id=inst_id)
            q_res = con.session.execute(q)
            aux_serial = q_res.current_rows[0]["obj_serial"]
            aux_serial = aux_serial[0] if len(aux_serial) > 0 else ''
            serialized_obj += aux_serial

    if len(serialized_obj) > 0:
        unserial_obj = pickle.loads(
            codecs.decode(serialized_obj.encode(), "base64"))
        return unserial_obj
    else:
        return None


def check_if_obj_exists(con, network_id, pt_id, obj_id, model_id):
    query = "select obj_serial from mv_objects " \
            "where pt_id='{pt_id}' " \
            "and model_id='{model_id}' " \
            "and obj_id='{obj_id}';"
    q = query.format(network_id=network_id, obj_id=obj_id, model_id=model_id,
                     pt_id=pt_id)
    q_res = con.session.execute(q)
    if len(q_res.current_rows[0]["obj_serial"]) > 0:
        return True
    else:
        # Search for partitioned object 'model_0':
        q = query.format(obj_id=obj_id, model_id=str(model_id) + '_' + str(0),
                         pt_id=pt_id)
        q_res = con.session.execute(q)
        if len(q_res.current_rows[0]["obj_serial"]) > 0:
            return True
        else:
            return False
