import copy
import os
import numpy as np

import sys
sys.path.append("../")
from src.utils import sql_utils

class staticTableHistogram(object):
    def __init__(self, initial_data_path, n_bins, min_vals, max_vals, attr_no_map, table_size_threshold):
        self.table_name = os.path.basename(initial_data_path)[0:-4]
        print(f'table = {self.table_name}')

        self.n_bins = n_bins
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.bin_sizes = (max_vals - min_vals) / self.n_bins
        self.attr_no_map = attr_no_map
        self.no_attr_map = {}
        for attr, no in self.attr_no_map.items():
            self.no_attr_map[no] = attr

        self.n_attrs = len(attr_no_map)
        # self.attr_type_list = attr_type_list
        arange_idxes = np.reshape(np.arange(0, self.n_attrs, dtype=np.int64), [1, -1])
        self.idxes = np.concatenate([arange_idxes, np.zeros(shape=arange_idxes.shape, dtype=arange_idxes.dtype)],
                                    axis=0)
        self.table_size_threshold = table_size_threshold

        assert initial_data_path is not None
        with open(initial_data_path, "r") as reader:
            lines = reader.readlines()
        lines = lines[1:]
        # self.table_size = len(lines)
        # self.histogram = np.zeros(shape=[self.n_attrs, self.n_bins], dtype=np.int64)
        _initial_data = []
        for k in range(self.n_attrs):
            _initial_data.append([])
        for line in lines:
            terms = line.split(",")
            for i, _term in enumerate(terms):
                term = _term.strip()
                if len(term) > 0:
                    val = float(term)
                    _initial_data[i].append(val)
        self.initial_data = [np.sort(np.array(data_i, dtype=np.float64)) for data_i in _initial_data]
        boundary_points = []
        for k in range(self.n_attrs):
            arange_idxes = np.reshape(np.arange(1, self.n_bins + 1, dtype=np.float64), [1, -1])
            boundary_points.append(arange_idxes)
        boundary_points = np.concatenate(boundary_points, axis=0)
        boundary_points *= np.reshape(self.bin_sizes, [-1, 1])
        assert (boundary_points.shape[0] == min_vals.shape[0])
        for i in range(min_vals.shape[0]):
            boundary_points[i] += min_vals[i]
        boundary_points[:, -1] += 1
        # print(f'boundary_points.shape = {boundary_points.shape}, len(lines) = {len(lines)}')
        self.boundary_points = boundary_points

        self.base_histogram = []
        for i in range(self.n_attrs):
            sorted_idxes = np.searchsorted(self.initial_data[i], boundary_points[i], side='left')
            print(f'\tattr = {self.no_attr_map[i]}')
            print(f'\t\tsorted_idxes[0:5] = {sorted_idxes[0:5]}')
            print(f'\t\tsorted_idxes[-5:] = {sorted_idxes[-5:]}')
            print(f'\t\tbounary_points[0:5] = {boundary_points[i][0:5]}')
            print(f'\t\tinitial_data[0:5] = {self.initial_data[i][0:5]}')
            assert sorted_idxes[-1] == self.initial_data[i].shape[0]
            sorted_idxes_left_translation = copy.deepcopy(sorted_idxes)
            sorted_idxes_left_translation[1:] = sorted_idxes_left_translation[0:-1]
            sorted_idxes_left_translation[0] = 0
            diff = sorted_idxes - sorted_idxes_left_translation
            self.base_histogram.append(np.reshape(diff, [1, -1]))
        self.base_histogram = np.concatenate(self.base_histogram, axis=0, dtype=np.float64)
        # print(f'table = {os.path.basename(initial_data_path)[0:-4]}, count = {np.sum(self.base_histogram, axis=1)}')

    # The format of a clause is like 'b.id > 1 and b.date > 100 and b.date <= 1000
    def calc_histogram_feature(self, filter_conds, print_flag=False):
        if print_flag:
            no_attr_map = {}
            for attr, no in self.attr_no_map.items():
                no_attr_map[no] = attr
            print(f'table = {self.table_name}')
            print(f'\tfilter_conds = {filter_conds}')



        if filter_conds is None or len(filter_conds) == 0:
            feature = np.reshape(self.base_histogram / self.table_size_threshold, [-1])
            return feature

        histogram = copy.deepcopy(self.base_histogram)
        filter_ranges = np.zeros(shape=[self.n_attrs, 2], dtype=np.float64)
        for i in range(self.n_attrs):
            attr_data = self.initial_data[i]
            filter_ranges[i][0] = attr_data[0]
            filter_ranges[i][1] = attr_data[-1] + 1

        relevant_attrs = set()
        for (attr_name, op, rhs) in filter_conds:
            i = self.attr_no_map[attr_name]

            if op == '<':
                filter_ranges[i][1] = float(rhs) - 0.5
            elif op == '<=':
                filter_ranges[i][1] = float(rhs) + 0.5
            if op == '>':
                filter_ranges[i][0] = float(rhs) + 0.5
            elif op == '>=':
                filter_ranges[i][0] = float(rhs) - 0.5
            else: # op == '=='
                filter_ranges[i][0] = float(rhs) - 0.5
                filter_ranges[i][1] = float(rhs) + 0.5
            relevant_attrs.add(i)

        for i in relevant_attrs:
            attr_data = self.initial_data[i]
            start_idx = np.searchsorted(attr_data, filter_ranges[i][0], side='left')
            start_idx = max(0, start_idx)
            end_idx = np.searchsorted(attr_data, filter_ranges[i][1], side='right')
            attr_data = attr_data[start_idx:end_idx]

            sorted_idxes = np.searchsorted(attr_data, self.boundary_points[i], side='left')
            sorted_idxes_left_translation = copy.deepcopy(sorted_idxes)
            sorted_idxes_left_translation[1:] = sorted_idxes_left_translation[0:-1]
            sorted_idxes_left_translation[0] = 0
            diff = sorted_idxes - sorted_idxes_left_translation
            histogram[i] = diff

            if print_flag:
                attr_name = no_attr_map[i]
                print(f'\tattr = {attr_name}')
                print(f'\thistogram = {histogram[i].astype(np.int64)}')
                print(f'\tbase_histogram = {self.base_histogram[i].astype(np.int64)}')
                print(f'\ttable_card = {np.sum(self.base_histogram[i].astype(np.int64))}')
        feature = np.reshape(histogram / self.table_size_threshold, [-1])
        return feature


class staticDBHistogram(object):
    def __init__(self, tables_info, data_dir, n_bins, checkpoint_dir):
        table_no_map, no_table_map, table_card_list, attr_no_map_list \
            , attr_no_types_list, attr_ranges_list = tables_info
        self.table_no_map = table_no_map

        self.no_table_map = no_table_map
        self.n_tables = len(attr_ranges_list)
        self.table_historgrams = []


        self.histogram_features = None
        self.num_inserts_before_queries = None
        self.train_idxes = None
        self.train_sub_idxes = None
        self.test_idxes = None
        self.test_sub_idxes = None
        self.split_idxes = None

        self.query_and_results = None

        for i in range(self.n_tables):
            table_name = self.no_table_map[i]
            fname = f'{table_name}.csv'
            path = os.path.join(data_dir, fname)
            th = staticTableHistogram(path,
                                      n_bins=n_bins,
                                      min_vals=attr_ranges_list[i][:, 0],
                                      max_vals=attr_ranges_list[i][:, 1],
                                      attr_no_map=attr_no_map_list[i],
                                      table_size_threshold=table_card_list[i])
            self.table_historgrams.append(th)
        print('-' * 50)


    def build_db_states(self, workload_results_path):
        with open(workload_results_path, "r") as reader:
            lines = reader.readlines()

        start_line_no = 0
        for i, line in enumerate(lines):
            if line.startswith('--'):# and line.find('part {0:d}'.format(self.start_from_part_no)) >= 0:
                start_line_no = i + 1
                break

        self.query_and_results = []
        j = 0

        db_states = []
        train_idxes = []
        train_sub_idxes = []
        test_idxes = []
        test_sub_idxes = []
        test_single_idxes = []

        all_queries = []

        for i in range(len(lines)):
            line = lines[i].strip().lower()
            if line.startswith("t"): # train or train_sub or test or test_sub
                if i >= start_line_no:
                    terms = line.split("||")
                    sql_part = terms[0].strip()
                    idx = sql_part.find(' ')
                    query = sql_part[idx + 1:].strip()
                    assert query.startswith('select')
                    if line.startswith('train_query'):
                        train_idxes.append(j)
                    elif line.startswith('train_sub'):
                        train_sub_idxes.append(j)
                    elif line.startswith('test_query'):
                        test_idxes.append(j)
                    elif line.startswith('test_sub'):
                        test_sub_idxes.append(j)
                    else:
                        # line.startswith('test_single')
                        assert line.startswith('test_single')
                        test_single_idxes.append(j)
                    all_queries.append(line)
                    # if print_flag:
                    #     print(f'query = {query}')

                    short_full_table_name_map, join_conds, filter_conds, analytic_functions = sql_utils.simple_parse_general_query(query)
                    filter_conds_list = []
                    for _ in range(self.n_tables):
                        filter_conds_list.append([])
                    for filter_cond in filter_conds:
                        (lhs, op, rhs) = filter_cond
                        terms = lhs.split('.')
                        short_table_name = terms[0]
                        attr_name = terms[1]
                        full_table_name = short_full_table_name_map[short_table_name]
                        table_id = self.table_no_map[full_table_name]
                        filter_conds_list[table_id].append((attr_name, op, rhs))
                    db_state = []
                    for table_id in range(self.n_tables):
                        db_state.append(
                            self.table_historgrams[table_id].calc_histogram_feature(filter_conds_list[table_id]) #, print_flag=print_flag)
                        )
                    db_state = np.concatenate(db_state, axis=0, dtype=np.float64)
                    db_state = np.reshape(db_state, [-1])
                    db_states.append(db_state)
                    j += 1

            if i % 10000 == 0:
                print(i, 'finished')
        db_states = np.array(db_states, dtype=np.float64)
        train_idxes = np.array(train_idxes, dtype=np.int64)
        train_sub_idxes = np.array(train_sub_idxes, dtype=np.int64)
        test_idxes = np.array(test_idxes, dtype=np.int64)
        test_sub_idxes = np.array(test_sub_idxes, dtype=np.int64)
        test_single_idxes = np.array(test_single_idxes, dtype=np.int64)
        return db_states, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes, all_queries
