import os
import sys
sys.path.append("../")
from src.utils import file_utils, sql_utils
from pathlib import Path
from src import config
from src.data_process import schema


def replace_term(path, old_term, new_term):
    lines = file_utils.read_all_lines(path)
    nlines = len(lines)
    for i in range(nlines):
        line = lines[i]
        if line.startswith('--'):
            break
        else:
            line = line.replace(old_term, new_term)
            lines[i] = line

    file_utils.write_all_lines(path, lines)


def generate_full_workload(simplified_workload_path, mysql_workload_path, table_attr_names_map):
    newlines = []
    lines = file_utils.read_all_lines(simplified_workload_path)

    start_i = -1
    for i, line in enumerate(lines):
        if line.startswith('--'):
            newlines.append(line)
            start_i = i + 1
            break
        newlines.append(line)

    lines = lines[start_i:]

    for line_no, line in enumerate(lines):
        # if line.startswith('test_query') or line.startswith('test_sub'):
        if line.startswith('t'):
            newlines.append(line)
        elif line.startswith('1|'):  # insert
            terms = line.strip().split('|')
            table_name = terms[1]
            values = terms[2]
            attr_names = table_attr_names_map[table_name]
            attrs_str = ','.join(attr_names)
            insert_sql = f'insert into {table_name} ({attrs_str}) values ({values});\n'
            newlines.append(insert_sql)
        elif line.startswith('2|'):  # delete
            terms = line.strip().split('|')
            table_name = terms[1]
            delete_cond = terms[2]
            values_str = terms[3]
            value_terms = values_str.split(',')
            attr_names = table_attr_names_map[table_name]
            assert len(attr_names) == len(value_terms)
            # terms = [attr_names[attr_no] + ' = ' + value_terms[attr_no] for attr_no in range(len(value_terms))]
            # delete_cond = ' and '.join(terms)
            delete_sql = f'delete quick from {table_name} where {delete_cond} limit 1;##{values_str}\n'
            newlines.append(delete_sql)
        elif line.startswith('3|'):  # update
            terms = line.strip().split('|')
            table_name = terms[1]
            delete_cond = terms[2]
            values_str = terms[3]
            terms = values_str.split('#')
            ins_vals_str = terms[1]
            insert_terms = ins_vals_str.split(',')
            attr_names = table_attr_names_map[table_name]
            attrs_str = ','.join(attr_names)
            insert_vals = ','.join(insert_terms)
            set_clause = f'({attrs_str}) = ({insert_vals})'
            update_sql = f'update {table_name} set {set_clause} where {delete_cond};##{values_str}\n'
            newlines.append(update_sql)
        else:
            newlines.append(line)

    file_utils.write_all_lines(mysql_workload_path, newlines)


def initialize(cfg):
    tables_info = schema.get_tables_info(cfg)
    table_attr_names_map = sql_utils.get_table_attr_names_map(tables_info)
    project_root = str(cfg.dataset.project_root)
    if project_root.endswith('/'):
        project_root = project_root[0:-1]

    wl_types = ['static', 'dynamic']
    for wl_type in wl_types:
        workload_dir = config.get_workload_dir(cfg, wl_type)
        simplified_workload_path = os.path.join(workload_dir, 'simplified_workload.sql')
        replace_term(simplified_workload_path, old_term='$PROJECT_ROOT$', new_term=project_root)
        mysql_workload_path = os.path.join(workload_dir, 'mysql_workload.sql')
        generate_full_workload(simplified_workload_path, mysql_workload_path, table_attr_names_map)



if __name__ == '__main__':
    cfg = config.getConfigs()
    initialize(cfg)

# python init/initialize.py
