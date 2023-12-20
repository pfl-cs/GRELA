import sys
import time
import re
from scipy.cluster.hierarchy import DisjointSet
from . import file_utils

def is_join_predicate(rhs, short_full_table_name_map):
    for short_name in short_full_table_name_map.keys():
        s = short_name + "."
        if rhs.startswith(s):
            return True
    return False

def predicates_parse(predicate):
    delim_pattern = '(?:(?:<|>)?=)|<|>'
    patt = re.compile(delim_pattern)
    items = re.split(delim_pattern, predicate)
    search_obj = patt.search(predicate)
    if search_obj is None:
        return None
    op = search_obj.group()

    lhs = items[0].strip()
    rhs = items[1].strip()
    return (lhs, op, rhs)


def simple_parse_count_query(_query):
    short_full_table_name_map, join_conds, filter_conds, analytic_functions = simple_parse_general_query(_query)
    assert len(analytic_functions) == 1
    return short_full_table_name_map, join_conds, filter_conds


def simple_parse_general_query(_query):
    query = _query.lower().strip()
    try:
        terms = query.split(' from ')
        l = len('select ')
        ana_funcs_str = terms[0][l:]
        ana_funcs_str = ana_funcs_str.strip()
        if ana_funcs_str.lower() == 'count(*)':
            # analytic_functions = None
            analytic_functions = [ana_funcs_str]
        else:
            analytic_functions = ana_funcs_str.split(', ')

        sql = terms[1].strip()
        sql = sql[0:-1]
    except:
        print('query =', query)
        raise Exception()

    sql_parts = sql.split(" where ")

    short_names = sql_parts[0].strip()
    terms = short_names.split(",")
    short_full_table_name_map = {}
    for term in terms:
        items = term.strip().split(" as ")

        if len(items) != 2:
            items = term.strip().split(" ")

        # assert (len(items) == 2)
        if len(items) == 2:
            full_name = items[0].strip()
            short_name = items[1].strip()
            short_full_table_name_map[short_name] = full_name
        else:
            short_full_table_name_map[term] = term


    if len(sql_parts) < 2:
        return short_full_table_name_map, [], [], analytic_functions

    predicates_str = sql_parts[1].strip()
    delim_pattern = '(?:(?:<|>)?=)|<|>'
    patt = re.compile(delim_pattern)

    predicates = predicates_str.split(" and ")

    join_conds = []
    filter_conds = []
    for predicate in predicates:
        items = re.split(delim_pattern, predicate)
        if len(items) != 2:
            print('query =', _query)
            print(len(items), items)
        assert (len(items) == 2)
        search_obj = patt.search(predicate)
        op = search_obj.group()

        lhs = items[0].strip()
        rhs = items[1].strip()

        if is_join_predicate(rhs, short_full_table_name_map):
            join_conds.append((lhs, op, rhs))
        else:
            filter_conds.append((lhs, op, rhs))

    return short_full_table_name_map, join_conds, filter_conds, analytic_functions


def num_involved_tables(_query):
    query = _query.lower()
    idx = query.find(' from ')
    s = query[idx + 6:]
    idx = s.find(' where ')
    s = s[0:idx].strip()
    terms = s.split(',')
    return len(terms)



def merge_elements_into_count_query(short_full_table_name_map, join_conds, filter_conds):
    return merge_elements_into_general_query(short_full_table_name_map, join_conds, filter_conds, None)

def merge_elements_into_general_query(short_full_table_name_map, join_conds, filter_conds, analytic_functions):
    if analytic_functions is None:
        query = 'select count(*) from '
    else:
        ana_funcs_str = ', '.join(analytic_functions)
        query = f'select {ana_funcs_str} from '

    sub_clauses = []
    for short_name in short_full_table_name_map:
        full_name = short_full_table_name_map[short_name]
        sub_clauses.append(full_name + ' as ' + short_name)
    sub_clauses.sort()
    query += ', '.join(sub_clauses)
    if len(join_conds) <= 0 and len(filter_conds) <= 0:
        return query + ';'
    # assert (len(join_conds) > 0 or len(filter_conds) > 0)
    query += ' where '

    if len(join_conds) > 0:
        sub_clauses = []
        equi_join_conds = []
        for join_cond in join_conds:
            (lhs, op, rhs) = join_cond
            if op == '=':
                equi_join_conds.append(join_cond)
            else:
                sub_clauses.append(lhs + ' ' + op + ' ' + rhs)
        equi_classes = get_equi_classes(equi_join_conds)
        sub_clauses.extend(equi_classes_to_join_predicates(equi_classes))

        sub_clauses.sort()
        query += ' and '.join(sub_clauses)
        if len(filter_conds) > 0:
            query += ' and '

    if len(filter_conds) > 0:
        sub_clauses = []
        for filter_cond in filter_conds:
            (lhs, op, rhs) = filter_cond
            sub_clauses.append(lhs + ' ' + op + ' ' + rhs)
        sub_clauses.sort()
        query += ' and '.join(sub_clauses)
    query += ';'

    return query


def get_join_query(query):
    short_full_table_name_map, join_conds, filter_conds, analytic_functions = simple_parse_general_query(query)
    if join_conds is None:
        return None
    else:
        return merge_elements_into_general_query(short_full_table_name_map, join_conds, [], analytic_functions)

def get_attr_ranges(table_attr, short_full_table_name_map, tables_info):
    terms = table_attr.strip().split('.')
    table = terms[0]
    attr = terms[1]
    table_no_map, no_table_map, table_card_list, attr_no_map_list \
        , attr_no_types_list, attr_ranges_list = tables_info

    full_name = short_full_table_name_map[table]
    table_no = table_no_map[full_name]
    attr_no = attr_no_map_list[table_no][attr]
    return attr_ranges_list[table_no][attr_no]

def get_equi_classes(join_conds):
    equi_classes = DisjointSet()
    for join_cond in join_conds:
        (lhs, op, rhs) = join_cond
        equi_classes.add(lhs)
        equi_classes.add(rhs)
        equi_classes.merge(lhs, rhs)
    return equi_classes


def get_query_equi_classes(query):
    short_full_table_name_map, join_conds, filter_conds, _ = simple_parse_general_query(query)

    if len(join_conds) == 0:
        return None
    else:
        try:
            return get_equi_classes(join_conds)
        except:
            print('-' * 30)
            print(f'query = {query}')
            for join_cond in join_conds:
                print(join_cond)
            print('-' * 30)
            raise Exception()


def format_join_clause(clause):
    terms = clause.split(' and ')
    new_terms = []
    for term in terms:
        items = term.split(' = ')
        assert len(items) == 2
        if items[0] > items[1]:
            new_terms.append(items[1] + ' = ' + items[0])
        else:
            new_terms.append(term)
    new_terms.sort()
    return ' and '.join(new_terms)


def table_and_attr(s):
    terms = s.split(".")
    return terms[0], terms[1]

def equi_classes_to_join_predicates(equi_classes):
    sub_clauses = []
    for sub_equi_class in equi_classes.subsets():
        table_attrs = list(sub_equi_class)
        table_attrs.sort()
        for j in range(1, len(table_attrs)):
            sub_clause = table_attrs[j - 1] + ' = ' + table_attrs[j]
            sub_clauses.append(sub_clause)
    return sub_clauses

def equi_class_list_to_join_clause(sub_equi_classes):
    sub_clauses = []
    for sub_equi_class in sub_equi_classes:
        table_attrs = list(sub_equi_class)
        table_attrs.sort()
        for j in range(1, len(table_attrs)):
            sub_clause = table_attrs[j - 1] + ' = ' + table_attrs[j]
            sub_clauses.append(sub_clause)
    clause = ' and '.join(sub_clauses)
    return clause

def load_queries(path):
    lines = file_utils.read_all_lines(path)
    queries = []
    for _line in lines:
        terms = _line.strip().split('||')
        query = terms[0].lower()
        queries.append(query)
    return queries

# create table movie_companies (movie_id integer not null, company_id integer not null, company_type_id integer not null);
def parse_create_sql(create_sql):
    terms = create_sql.strip().lower().split('create table')
    info = terms[1].strip()
    lidx = info.find('(')
    table_name = info[0:lidx].strip()
    ridx = info.rfind(')')
    attr_infos_str = info[lidx + 1: ridx].strip()
    attr_infos = attr_infos_str.split(',')

    attr_descs = []
    attr_extra_infos = []
    for attr_info in attr_infos:
        terms = attr_info.strip().split(' ')
        attr_name = terms[0].strip()
        data_type = terms[1].strip()
        extra_info = None
        if len(terms) > 2:
            extra_info = ' '.join(terms[2:])
        attr_extra_infos.append(extra_info)
        assert data_type in {'bigint', 'integer', 'character', 'double', 'smallint', 'timestamp', 'serial'}
        if data_type == 'character':
            varying_str = terms[2].strip()
            assert (varying_str.startswith('varying'))
        # assert (data_type == 'integer')
        attr_descs.append((attr_name, data_type))
    return table_name, attr_descs, attr_extra_infos

def get_attr_infos_from_create_sql(create_sql):
    table_name, attr_descs, attr_extra_infos = parse_create_sql(create_sql)
    attr_type_list = []
    attr_names = []
    for attr_desc in attr_descs:
        attr_name = attr_desc[0]
        attr_type = attr_desc[1]
        attr_names.append(attr_name)
        if attr_type in {'bigint', 'integer', 'smallint', 'serial'}:
            attr_type_list.append(0)
        elif attr_type == 'double':
            attr_type_list.append(1)
        else:
            attr_type_list.append(-1)

    return table_name, attr_names, attr_type_list

def get_all_table_attr_infos(create_tables_path):
    lines = file_utils.read_all_lines(create_tables_path)
    table_attr_infos_list = []
    table_attr_types_map = {}
    for _line in lines:
        line = _line.lower()
        if line.startswith('create table'):
            table_name, attr_names, attr_type_list = get_attr_infos_from_create_sql(line.strip())
            table_attr_infos_list.append((table_name, attr_names, attr_type_list))
            table_attr_types_map[table_name] = attr_type_list
    return table_attr_infos_list, table_attr_types_map


def format_count_query_alphabetically(sql):
    short_full_table_name_map, join_conds, filter_conds = simple_parse_count_query(sql)
    return merge_elements_into_count_query(short_full_table_name_map, join_conds, filter_conds)


def get_table_attr_names_map(tables_info):
    (table_no_map, no_table_map, table_card_list, attr_no_map_list, attr_no_types_list, attr_ranges_list) = tables_info
    table_attr_names_map = {}
    for table_name in table_no_map:
        table_no = table_no_map[table_name]
        attr_no_map = attr_no_map_list[table_no]
        attr_names = [None] * len(attr_no_map)
        for attr_name in attr_no_map:
            attr_no = attr_no_map[attr_name]
            attr_names[attr_no] = attr_name
        table_attr_names_map[table_name] = attr_names

    return table_attr_names_map

def add_analytic_functions(sql, table_attr_names_map):
    short_full_table_name_map, join_conds, filter_conds, analytic_functions = simple_parse_general_query(sql)
    assert len(analytic_functions) == 1
    # assert analytic_functions is None

    func_types = ['avg({0:s})', 'sum({0:s})', 'min({0:s})', 'max({0:s})']
    analytic_functions = ['count(*)']
    for short_name in short_full_table_name_map:
        full_table_name = short_full_table_name_map[short_name]
        table_attrs = table_attr_names_map[full_table_name]
        for _table_attr in table_attrs:
            table_attr = f'{short_name}.{_table_attr}'
            for func_type in func_types:
                analytic_functions.append(func_type.format(table_attr))
    return merge_elements_into_general_query(short_full_table_name_map, join_conds, filter_conds, analytic_functions)

def merge_join_conds(join_conds):
    equi_join_conds = []
    for join_cond in join_conds:
        (lhs, op, rhs) = join_cond
        assert op == '='
        equi_join_conds.append(join_cond)
    equi_classes = get_equi_classes(equi_join_conds)
    sub_clauses = equi_classes_to_join_predicates(equi_classes)

    sub_clauses.sort()
    s = ' and '.join(sub_clauses)
    return s

def split_count_query_into_single(query):
    short_full_table_name_map, join_conds, filter_conds = simple_parse_count_query(query)
    template = 'select count(*) from {0:s} as {1:s} where {2:s};'
    attr_conds_map = {}
    for (lhs, op, rhs) in filter_conds:
        assert op == '='
        short_name, attr_name = table_and_attr(lhs)
        if short_name in attr_conds_map:
            attr_conds_map[short_name].append(f'{lhs} = {rhs}')
        else:
            attr_conds_map[short_name] = [f'{lhs} = {rhs}']
    single_queries = []
    for short_name in short_full_table_name_map:
        assert short_name in attr_conds_map
        conds = attr_conds_map[short_name]
        assert len(conds) > 0
        full_name = short_full_table_name_map[short_name]
        predicates = ' and '.join(conds)
        q = template.format(full_name, short_name, predicates)
        single_queries.append(q)

    return single_queries

def get_all_table_name_map(queries):
    short_full_table_name_map = {}
    full_short_table_name_map = {}
    for q in queries:
        try:
            _short_full_table_name_map, join_conds, filter_conds, analytic_functions = simple_parse_general_query(q)
        except:
            print('q =', q)
            raise Exception()

        for short_name in _short_full_table_name_map:
            full_name = _short_full_table_name_map[short_name]
            if short_name in short_full_table_name_map:
                assert short_full_table_name_map[short_name] == full_name
            else:
                short_full_table_name_map[short_name] = full_name

            if full_name in full_short_table_name_map:
                assert full_short_table_name_map[full_name] == short_name
            else:
                full_short_table_name_map[full_name] = short_name
    return short_full_table_name_map, full_short_table_name_map

def get_query_and_task_values_from_line(line):
    terms = line.strip().split('||')
    task_values_str = None
    if len(terms) == 3:
        task_values_str = terms[1]
        if task_values_str.startswith('-'):
            task_values_str = None
    query = terms[0]
    tag = None
    if not query.startswith('select'):
        idx = query.find(' ')
        tag = query[:idx+1]
        query = query[idx + 1:]
        if not query.startswith('select'):
            print('-' * 50)
            print(line)
            print(query)
            print('-' * 50)
        assert query.startswith('select')
    return tag, query, task_values_str


if __name__ == '__main__':
    # sql = 'select  count(*) from posts as p,  		postLinks as pl,          postHistory as ph,          votes as v,          badges as b,          users as u  where p.Id = pl.RelatedPostId 	and u.Id = p.OwnerUserId 	and u.Id = b.UserId 	and u.Id = ph.UserId     and u.Id = v.UserId  AND b.Date>=\'2010-07-20 15:54:11\'::timestamp  AND b.Date<=\'2014-09-11 14:01:23\'::timestamp  AND ph.CreationDate>=\'2011-02-05 21:42:45\'::timestamp  AND ph.CreationDate<=\'2014-07-23 07:28:49\'::timestamp  AND p.ViewCount<=2058  AND p.AnswerCount>=0  AND p.CommentCount>=0  AND p.CommentCount<=20  AND p.FavoriteCount<=36  AND u.Views>=0  AND u.DownVotes>=0  AND u.UpVotes>=0  AND v.VoteTypeId=2;'
    # sql = timestamp_to_int_in_stats_sql(sql)
    pass
