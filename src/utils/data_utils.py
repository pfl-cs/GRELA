from . import file_utils
import numpy as np
import random
import pandas as pd
import time
import os

def array_to_appear_times_map(data):
    val_appear_times_map = {}
    count = 1
    last_val = data[0]
    for i in range(1, data.shape[0]):
        val = data[i]
        if val == last_val:
            count += 1
        else:
            val_appear_times_map[last_val] = count
            last_val = val
            count = 1
    val_appear_times_map[last_val] = count
    return val_appear_times_map

def generate_random_idxes(N):
    idxes = np.arange(0, N, dtype=np.int64).tolist()
    random.shuffle(idxes)
    idxes = np.array(idxes, dtype=np.int64)
    return idxes

def save_to_csv_file(data, attr_names, dst):
    lines = [','.join(attr_names) + '\n']
    n = data[0].shape[0]
    assert len(data) == len(attr_names)

    for j in range(n):
        lines.append(','.join([str(x[j]) for x in data]) + '\n')
    file_utils.write_all_lines(dst, lines)

def read_csv_file(data_path, attr_type_list):
    try:
        # data = pd.read_csv(data_path, sep=',', header=None)
        _data = pd.read_csv(data_path, sep=',')
    except:
        print('data_path =', data_path)
        raise Exception()

    attr_names = _data.columns.tolist()
    assert len(attr_names) == len(attr_type_list)
    numpy_types = [np.int64, np.float64]
    attr_numpy_types = [numpy_types[x] for x in attr_type_list]
    try:
        data = [_data[attr_name] for attr_name in attr_names]
        np_data = []
        for i, data_i in enumerate(data):
            dtype = attr_numpy_types[i]
            check_for_nan = data_i.isnull().values.any()
            assert check_for_nan == False
            # try:
            #     dtype = attr_numpy_types[i]
            # except:
            #     print('i =', i)
            #     print('len(num_types) =', len(numpy_types))
            #     print('len(attr_names) =', len(attr_names))
            #     print('len(data) =', len(data))
            #     raise Exception()
            if dtype == np.int64:
                try:
                    min_value = min(int(data_i.min() - 1.5), -1)
                    data_i += 0.5
                    data_i = data_i // 1
                    x = data_i.to_numpy(copy=True, dtype=np.int64, na_value=min_value)
                except:
                    print('attr_name =', attr_names[i])
                    print('data_i.dtype=', data_i.dtype)
            else:
                min_value = min(float(data_i.min() - 1), -1)
                x = data_i.to_numpy(copy=True, dtype=np.float64, na_value=min_value)
            np_data.append(x)

        # np_data = [_data[attr_name].to_numpy(dtype=attr_numpy_types[i], copy=True) for i, attr_name in
        #         enumerate(attr_names)]
    except:
        print('attr_names =', attr_names)
        print('attr_numpy_types =', attr_numpy_types)
        raise Exception()
    table_size = data[0].shape[0]
    return attr_names, np_data, table_size

def read_csv_and_detect_attr_types(data_path):
    try:
        # data = pd.read_csv(data_path, sep=',', header=None)
        _data = pd.read_csv(data_path, sep=',')
    except:
        print('data_path =', data_path)
        raise Exception()

    attr_names = _data.columns.tolist()
    attr_type_list = []
    try:
        data = [_data[attr_name] for attr_name in attr_names]
        for i, data_i in enumerate(data):
            min_value = data_i.min() - 1
            x = data_i.to_numpy(copy=True, na_value=min_value)
            if np.issubdtype(x.dtype, np.integer):
                data[i] = x.astype(np.int64)
                attr_type_list.append(0)
            else:
                if not np.issubdtype(x.dtype, np.float64):
                    print('+++++x.dtype =', x.dtype)
                assert np.issubdtype(x.dtype, np.float64)
                data[i] = x.astype(np.float64)
                attr_type_list.append(1)
    except:
        raise Exception()
    table_size = data[0].shape[0]
    return attr_names, attr_type_list, data, table_size

def average_split(data, num_parts):
    """
    :param data: list
    :param num_parts:
    :return:
    """
    assert len(data) % num_parts == 0
    n = len(data) // num_parts
    data_list = []
    cursor = 0
    for i in range(n):
        data_list.append(data[cursor: cursor + n])
        cursor += n
    return data_list


def time_to_int(s):
    timestamp = time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
    timestamp = int(timestamp + 0.5)
    return timestamp

def process_empty_values(src, dst, attr_type_list):
    attr_names, data, table_size = read_csv_file(src, attr_type_list)
    assert len(attr_names) == len(attr_type_list)
    for i, attr_type in enumerate(attr_type_list):
        if attr_type != 0:
            file_name = os.path.basename(src)
            table_name = file_name[0:-4]
            print('table_name = {0:s}, attr_name = {1:s}, attr_type = {2:d}'.format(table_name, attr_names[i], attr_type))
        assert attr_type == 0
    save_to_csv_file(data, attr_names, dst)
