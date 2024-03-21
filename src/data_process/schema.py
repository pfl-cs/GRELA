import numpy as np
import os
import sys
sys.path.append("../..")
from src.utils import sql_utils, file_utils
epsilon_for_float = 1e-6


def get_table_info(table_path, attr_type_list):
    attr_no_map = {}
    attr_no_types = []
    lines = file_utils.read_all_lines(table_path)

    attr_names_str = lines[0].strip()
    attr_names = attr_names_str.split(",")
    for i, attr_name in enumerate(attr_names):
        attr_no_map[attr_name.lower()] = i

    assert attr_type_list is not None
    assert len(attr_type_list) == len(attr_names)

    epsilons = []
    for attr_type in attr_type_list:
        assert attr_type == 0 or attr_type == 1
        if attr_type == 0:
            epsilons.append(0.5)
        else:
            epsilons.append(epsilon_for_float)
        attr_no_types.append(0)  # all attrs' types are int

    attr_no_types = np.array(attr_type_list, dtype=np.int64)

    lines = lines[1:]
    card = len(lines)

    data = []
    for i in range(len(attr_names)):
        data.append([])
    for line in lines:
        items = line.strip().split(",")
        try:
            for i, x in enumerate(items):
                if len(x) > 0:
                    data[i].append(float(x))
        except:
            print('-' * 20 + line)
            for i, x in enumerate(items):
                print('i =', i, 'x =', x, 'len(x) =', len(x), )
                if len(x) > 0:
                    data[i].append(float(x))
            raise

    table_attr_ranges = []
    data_0 = np.array(data[0], dtype=np.float64)
    data_1 = np.array(data[1], dtype=np.float64)
    diff = np.abs(data_1 - data_0)
    print('table =', os.path.basename(table_path)[0:-4], ', diff.max =', np.max(diff), 'diff.shape =', diff.shape, '-' * 10)
    for i, data_i in enumerate(data):
        data_i_np = np.array(data_i, dtype=np.float64)
        minv, maxv = np.min(data_i_np) - epsilons[i], np.max(data_i_np) + epsilons[i]
        table_attr_ranges.append([minv, maxv])
    table_attr_ranges = np.array(table_attr_ranges, dtype=np.float64)

    return attr_no_map, attr_no_types, table_attr_ranges, card


# skip check
def parse_str_int_int_map(line, item_sep="|", key_value_sep=","):
    terms = line.strip().split(item_sep)
    m1 = {}
    m2 = {}
    for term in terms:
        items = term.strip().split(key_value_sep)
        no = int(items[1])
        m1[items[0].strip()] = no
        m2[no] = int(items[2])
    return m1, m2

# skip check
def parse_str_int_map(line, item_sep="|", key_value_sep=","):
    terms = line.strip().split(item_sep)
    m = {}
    for term in terms:
        items = term.strip().split(key_value_sep)
        m[items[0].strip()] = int(items[1])
    return m


def load_tables_info_from_file(table_names, tables_info_path):
    attr_no_map_list = []
    attr_no_types_list = []
    attr_ranges_list = []
    lines = file_utils.read_all_lines(tables_info_path)

    table_no_map, table_card_map = parse_str_int_int_map(lines[0])
    n_tables = len(table_no_map)
    table_card_list = []
    for i in range(n_tables):
        table_card_list.append(table_card_map[i])

    lines = lines[3:]
    for i in range(n_tables):
        line = lines[i]
        terms = line.split(":")
        line = terms[1]
        attr_no_map = parse_str_int_map(line)
        attr_no_map_list.append(attr_no_map)

    lines = lines[n_tables + 1:]
    for i in range(n_tables):
        line = lines[i]
        terms = line.split(":")
        line = terms[1].strip()
        terms = line.split(",")
        attr_no_types = [int(x) for x in terms]
        attr_no_types = np.array(attr_no_types, dtype=np.int64)
        attr_no_types_list.append(attr_no_types)

    lines = lines[n_tables + 1:]
    for i in range(n_tables):
        line = lines[i]
        terms = line.split(":")
        line = terms[1].strip()
        terms = line.split(",")
        attr_ranges = [float(x) for x in terms]
        attr_ranges = np.array(attr_ranges, dtype=np.float64)
        attr_ranges = np.reshape(attr_ranges, [-1, 2])
        attr_ranges_list.append(attr_ranges)

    no_table_map = {}
    for table_name in table_names:
        no = table_no_map[table_name.lower()]
        no_table_map[no] = table_name
    return (table_no_map, no_table_map, table_card_list, attr_no_map_list, attr_no_types_list, attr_ranges_list)


def get_tables_info(cfg):
    create_tables_path = cfg.dataset.create_tables_path
    _, table_attr_types_map = sql_utils.get_all_table_attr_infos(create_tables_path)

    table_names = table_attr_types_map.keys()
    table_names = list(table_names)
    table_names.sort()
    table_card_list = []

    tables_info_path = cfg.dataset.tables_info_path

    attr_no_map_list = []
    attr_no_types_list = []
    attr_ranges_list = []
    if os.path.exists(tables_info_path) == False:
        table_no_map = {}
        no_keep_cap_letter_table_name_map = {}
        for i, table_name in enumerate(table_names):
            table_no_map[table_name.lower()] = i
            no_keep_cap_letter_table_name_map[i] = table_name

        for table_name in table_names:
            attr_type_list = table_attr_types_map[table_name]
            table_file_path = os.path.join(cfg.dataset.data_dir, table_name + ".csv")
            attr_no_map, attr_no_types, table_attr_ranges, table_card = get_table_info(table_file_path, attr_type_list)
            attr_no_map_list.append(attr_no_map)
            attr_no_types_list.append(attr_no_types)
            attr_ranges_list.append(table_attr_ranges)
            table_card_list.append(table_card)

        with open(tables_info_path, "w") as writer:
            lines = []
            terms = []
            terms2 = []

            # table_no_map
            for table_name, table_no in table_no_map.items():
                terms.append(table_name + "," + str(table_no) + "," + str(table_card_list[table_no]))
                terms2.append(
                    table_name + "," + str(table_no) + "," + no_keep_cap_letter_table_name_map[table_no] + "," + str(
                        table_card_list[table_no]))
            lines.append('|'.join(terms) + "\n")
            lines.append('|'.join(terms2) + "\n")
            lines.append("\n")

            # attr_no_map_list
            for i, attr_no_map in enumerate(attr_no_map_list):
                terms = []

                for attr_name, attr_no in attr_no_map.items():
                    terms.append(attr_name + "," + str(attr_no))
                line = table_names[i] + ' attr nos: ' + '|'.join(terms) + "\n"
                lines.append(line)
            lines.append("\n")

            # attr_no_type_map_list
            for i, attr_no_types in enumerate(attr_no_types_list):
                terms = attr_no_types.tolist()
                terms = [str(x) for x in terms]

                line = table_names[i] + ' attr types: ' + ','.join(terms) + "\n"

                lines.append(line)
            lines.append("\n")

            # table_attr_ranges_list
            for i, attr_ranges in enumerate(attr_ranges_list):
                print("+++++i =", i, attr_ranges)
                tmp = np.reshape(attr_ranges, [-1])
                terms = tmp.tolist()
                terms = [str(x) for x in terms]
                line = table_names[i] + ' attr ranges: ' + ','.join(terms) + "\n"
                lines.append(line)

            writer.writelines(lines)

    return load_tables_info_from_file(table_names, tables_info_path)
