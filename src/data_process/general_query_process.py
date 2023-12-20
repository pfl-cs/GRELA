import numpy as np
import sys
sys.path.append("../..")
from src.utils import sql_utils, file_utils, data_utils


epsilon_for_float = 1e-6

def isnumber(s):
    return s.lstrip('-').replace('.', '', 1).isdigit()

def get_table_no_and_attr_no(table_attr, table_no_map, attr_no_map_list):
    table_name, attr_name = sql_utils.table_and_attr(table_attr)
    table_no = table_no_map[table_name]
    attr_no_map = attr_no_map_list[table_no]
    assert attr_name in attr_no_map
    attr_no = attr_no_map[attr_name]
    return table_no, attr_no


def parse_predicate_lhs(lhs, table_no_map, attr_no_map_list):
    lhs = lhs.lower()
    idx = lhs.find("::timestamp")
    template = "2000-01-01 00:00:00"
    if idx >= 0:
        assert idx >= len(template) + 2
        lhs = lhs[0:idx - 1].strip()
        lhs = lhs[1:]
        timestamp = data_utils.time_to_int(lhs)
        return (-1, timestamp)
    terms = lhs.split(".")
    if len(terms) == 2:
        table_name = terms[0]
        attr = terms[1]
        if table_name in table_no_map:
            table_no = table_no_map[table_name]
            attr_no_map = attr_no_map_list[table_no]
            assert attr in attr_no_map
            attr_no = attr_no_map[attr]
            return (table_no, attr_no)
        else:
            assert isnumber(lhs)
            return (-1, float(lhs))
    else:
        if not isnumber(lhs):
            print('lhs =', lhs)
            raise Exception()
        assert isnumber(lhs)
        return (-1, float(lhs))


def parse_sql(sql_and_other_info,
              table_no_map,
              attr_no_map_list,
              attr_no_types_list,
              attr_ranges_list,
              delim,
              task_values_idx=1):
    terms = sql_and_other_info.split(delim)
    sql = terms[0].strip()
    task_values = None
    if len(terms) > task_values_idx:
        task_values_str = terms[task_values_idx].strip()
        value_terms = task_values_str.split(',')
        card = int(value_terms[0])
        if card > 0:
            task_values = [float(x) for x in value_terms]
            task_values[0] = card
        elif card == 0:
            task_values = [0]
        else:
            task_values = None

    idx = sql.find('select')
    assert (idx >= 0)
    query = sql[idx:]

    short_full_table_name_map, join_predicates, filter_predicates, analytic_functions = sql_utils.simple_parse_general_query(query)
    relevant_tables = []
    for short_name in short_full_table_name_map:
        full_name = short_full_table_name_map[short_name]
        table_no = table_no_map[full_name]
        table_no_map[short_name] = table_no
        relevant_tables.append(table_no)
    relevant_tables.sort()

    equi_classes = sql_utils.get_equi_classes(join_predicates)

    join_strs = []
    for equi_class in equi_classes.subsets():
        table_attr_list = list(equi_class)
        # table_attr_list.sort()
        for i, l_table_attr in enumerate(table_attr_list):
            for j, r_table_attr in enumerate(table_attr_list):
                if i != j:
                    join_strs.append(l_table_attr + ' = ' + r_table_attr)

    join_strs.sort()
    join_conds = []
    for join_str in join_strs:
        terms = join_str.split(' = ')
        l_table_attr = terms[0].strip()
        r_table_attr = terms[1].strip()
        l_table_no, l_attr_no = get_table_no_and_attr_no(l_table_attr, table_no_map, attr_no_map_list)
        r_table_no, r_attr_no = get_table_no_and_attr_no(r_table_attr, table_no_map, attr_no_map_list)
        join_conds.append([l_table_no, l_attr_no, r_table_no, r_attr_no])

    if len(join_conds) == 0:
        join_conds = None
    else:
        join_conds = np.array(join_conds, dtype=np.int64)

    attr_range_conds = [x.copy() for x in attr_ranges_list]
    filter_conds = []
    for filter_predicate in filter_predicates:
        (lhs, op, rhs) = filter_predicate
        (lhs_table_no, lhs_attr_no) = parse_predicate_lhs(lhs, table_no_map, attr_no_map_list)
        rhs_res = None
        try:
            rhs_res = parse_predicate_lhs(rhs, table_no_map, attr_no_map_list)
        except:
            print('filter_predicate =', filter_predicate)
            print('sql =', sql_and_other_info)
            raise Exception()


        assert (rhs_res[0] < 0)
        # Note I skip border check here. I will add it in the future
        attr_type = attr_no_types_list[lhs_table_no][lhs_attr_no]  # 0 for int, 1 for float
        rhs_val = rhs_res[1]
        single_attr_range_cond = attr_range_conds[lhs_table_no]
        if attr_type == 0:
            epsilon = 0.5
        else:
            epsilon = epsilon_for_float
        if op == '=':
            single_attr_range_cond[lhs_attr_no][0] = rhs_val - epsilon
            single_attr_range_cond[lhs_attr_no][1] = rhs_val + epsilon
        elif op == '<=':
            single_attr_range_cond[lhs_attr_no][1] = rhs_val + epsilon
        elif op == '<':
            single_attr_range_cond[lhs_attr_no][1] = rhs_val - epsilon
        elif op == '>=':
            single_attr_range_cond[lhs_attr_no][0] = rhs_val - epsilon
        else:  # '>'
            single_attr_range_cond[lhs_attr_no][0] = rhs_val + epsilon

    for table_no in relevant_tables:
        filter_conds.append(attr_range_conds[table_no])

    assert len(relevant_tables) > 0
    for x in filter_conds:
        assert len(x.shape) == 2
        assert x.shape[0] > 0
    filter_conds = np.concatenate(filter_conds, axis=0)

    attr_range_conds = np.concatenate(attr_range_conds, axis=0, dtype=np.float64)
    attr_range_conds = np.reshape(attr_range_conds, [-1])
    assert join_conds is None or (join_conds.shape[0] > 0)

    try:
        new_analytic_functions = [analytic_functions[0]]
    except:
        print('-' * 50)
        print(sql_and_other_info)
        print(analytic_functions)
        print(len(analytic_functions))
        print('-' * 50)
    analytic_functions = analytic_functions[1:]
    for func in analytic_functions:
        idx = func.find('(')
        short_table_attr = func[idx+1:-1]
        terms = short_table_attr.split('.')
        full_name = short_full_table_name_map[terms[0]]
        new_func = f'{func[0:idx]}({full_name}.{terms[1]})'
        new_analytic_functions.append(new_func)

    return join_conds, equi_classes, attr_range_conds, relevant_tables, filter_conds, new_analytic_functions, task_values


def parse_queries(lines, table_no_map, attr_no_map_list, attr_no_types_list, attr_ranges_list, delim):
    join_conds_list = []
    equi_classes_list = []
    attr_range_conds_list = []

    relevant_tables_list = []
    filter_conds_list = []
    analytic_functions_list = []
    task_values_list = []

    for line_no, line in enumerate(lines):

        join_conds, equi_classes, attr_range_conds, relevant_tables, filter_conds, analytic_functions, task_values = parse_sql(
            line,
            table_no_map,
            attr_no_map_list,
            attr_no_types_list,
            attr_ranges_list,
            delim
        )
        # if line_no == 109666:
        #     print('-' * 50)
        #     print(line)
        #     print(task_values)
        #     print('-' * 50)
        equi_classes_list.append(equi_classes)
        join_conds_list.append(join_conds)
        attr_range_conds_list.append(attr_range_conds)
        relevant_tables_list.append(relevant_tables)
        filter_conds_list.append(filter_conds)
        analytic_functions_list.append(analytic_functions)
        task_values_list.append(task_values)

    possible_join_strs = []
    for equi_classes in equi_classes_list:
        for equi_class in equi_classes.subsets():
            table_attr_list = list(equi_class)
            table_attr_list.sort()
            for i, l_table_attr in enumerate(table_attr_list):
                for j in range(i + 1, len(table_attr_list)):
                    r_table_attr = table_attr_list[j]
                    possible_join_strs.append(l_table_attr + ' = ' + r_table_attr)

    possible_join_strs = set(possible_join_strs)
    possible_join_strs = list(possible_join_strs)
    possible_join_strs.sort()

    possible_join_attrs = []
    for join_str in possible_join_strs:
        terms = join_str.split(' = ')
        l_table_attr = terms[0].strip()
        r_table_attr = terms[1].strip()
        l_table_no, l_attr_no = get_table_no_and_attr_no(l_table_attr, table_no_map, attr_no_map_list)
        r_table_no, r_attr_no = get_table_no_and_attr_no(r_table_attr, table_no_map, attr_no_map_list)

        possible_join_attrs.append([l_table_no, l_attr_no, r_table_no, r_attr_no])

    possible_join_attrs = np.array(possible_join_attrs, dtype=np.int64)

    return (possible_join_attrs, join_conds_list, attr_range_conds_list, relevant_tables_list, filter_conds_list,
            analytic_functions_list, task_values_list)

