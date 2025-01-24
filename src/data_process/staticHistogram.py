import copy
import os
import numpy as np

import sys
sys.path.append("../")
from src.utils import sql_utils, FileViewer

class staticTableHistogram(object):
    def __init__(self, initial_data_path, n_bins, min_vals, max_vals, attr_no_map, table_size_threshold, base_histogram_dir, keep_checkpoint=False):

        self.table_name = os.path.basename(initial_data_path)[0:-4]
        print(f'\t    Start to build the base histogram for {self.table_name}...')
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
        # self.idxes = np.concatenate([arange_idxes, np.zeros(shape=arange_idxes.shape, dtype=arange_idxes.dtype)],
        #                             axis=0)
        self.table_size_threshold = table_size_threshold

        if self.checkpoint_exists(base_histogram_dir):
            self.load_base_histogram(base_histogram_dir)
            # self.base_sorted_idxes = []
            # for i in range(self.n_attrs):
            #     data_i = self.initial_data[i]
            #     sorted_idxes = np.searchsorted(data_i, self.boundary_points[i], side='left')
            #     if sorted_idxes[-1] != data_i.shape[0]:
            #         print(f'table_name = {self.table_name}, sorted_idxes[-1] = {sorted_idxes[-1]}, data_i.shape = {data_i.shape}')
            #     assert sorted_idxes[-1] == data_i.shape[0]
            #     self.base_sorted_idxes.append(np.reshape(sorted_idxes, [1, -1]))
            # self.base_sorted_idxes = np.concatenate(self.base_sorted_idxes, axis=0, dtype=np.int64)
            # dir = os.path.join(base_histogram_dir, self.table_name)
            # base_sorted_idxes_path = os.path.join(dir, "base_sorted_idxes.npy")
            # np.save(base_sorted_idxes_path, self.base_sorted_idxes)
            return

        assert initial_data_path is not None
        with open(initial_data_path, "r") as reader:
            lines = reader.readlines()
        lines = lines[1:]
        # self.table_size = len(lines)
        # self.histogram = np.zeros(shape=[self.n_attrs, self.n_bins], dtype=np.int64)
        _initial_data = []
        for k in range(self.n_attrs):
            _initial_data.append([])
        for line_no, line in enumerate(lines):
            terms = line.split(",")
            for i, _term in enumerate(terms):
                term = _term.strip()
                if len(term) > 0:
                    val = float(term)
                    _initial_data[i].append(val)
            # if line_no % 1000000 == 0:
            #     print(f'\t{line_no} finished.')

        del lines
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
        self.boundary_points = boundary_points

        # self.initial_data = [np.sort(np.array(data_i, dtype=np.float64)) for data_i in _initial_data]

        self.base_histogram = []
        self.base_sorted_idxes = []
        self.initial_data = []

        dir = os.path.join(base_histogram_dir, self.table_name)
        if keep_checkpoint:
            FileViewer.detect_and_create_dir(dir)
        for i in range(self.n_attrs):
            # if i in self.no_attr_map:
            #     print(f'\tStart to processing {self.no_attr_map[i]}...')
            # else:
            #     print(f'\tStart to processing data-{i}...')
            data_i = np.array(_initial_data[i], dtype=np.float64)
            # tmp = _initial_data[i]
            _initial_data[i] = None
            # del tmp
            data_i = np.sort(data_i)
            self.initial_data.append(data_i)

            sorted_idxes = np.searchsorted(data_i, boundary_points[i], side='left')
            # print(f'\tattr = {self.no_attr_map[i]}')
            # print(f'\t\tsorted_idxes[0:5] = {sorted_idxes[0:5]}')
            # print(f'\t\tsorted_idxes[-5:] = {sorted_idxes[-5:]}')
            # print(f'\t\tbounary_points[0:5] = {boundary_points[i][0:5]}')
            # print(f'\t\tinitial_data[0:5] = {self.initial_data[i][0:5]}')
            if sorted_idxes[-1] != data_i.shape[0]:
                print(
                    f'table_name = {self.table_name}, sorted_idxes[-1] = {sorted_idxes[-1]}, data_i.shape = {data_i.shape}')
            assert sorted_idxes[-1] == data_i.shape[0]
            sorted_idxes_left_translation = copy.deepcopy(sorted_idxes)
            sorted_idxes_left_translation[1:] = sorted_idxes_left_translation[0:-1]
            sorted_idxes_left_translation[0] = 0
            diff = sorted_idxes - sorted_idxes_left_translation
            self.base_histogram.append(np.reshape(diff, [1, -1]))
            self.base_sorted_idxes.append(np.reshape(sorted_idxes, [1, -1]))

            if keep_checkpoint:
                initial_data_path = os.path.join(dir, f'data_{i}.npy')
                np.save(initial_data_path, data_i)

        self.base_histogram = np.concatenate(self.base_histogram, axis=0, dtype=np.float64)
        self.base_sorted_idxes = np.concatenate(self.base_sorted_idxes, axis=0, dtype=np.int64)
        # self.dump_base_histogram(base_histogram_dir)

        # for i in range(self.n_attrs):
        #     attr_data = self.initial_data[i]
        #     sorted_idxes = np.searchsorted(attr_data, self.boundary_points[i], side='left')
        #     sorted_idxes_left_translation = copy.deepcopy(sorted_idxes)
        #     sorted_idxes_left_translation[1:] = sorted_idxes_left_translation[0:-1]
        #     sorted_idxes_left_translation[0] = 0
        #     diff = sorted_idxes - sorted_idxes_left_translation
        #     histogram[i] = diff


        if keep_checkpoint:
            boundary_points_path = os.path.join(dir, 'boundary_points.npy')
            np.save(boundary_points_path, self.boundary_points)
            base_histogram_path = os.path.join(dir, "base_histogram.npy")
            np.save(base_histogram_path, self.base_histogram)
            base_sorted_idxes_path = os.path.join(dir, "base_sorted_idxes.npy")
            np.save(base_sorted_idxes_path, self.base_sorted_idxes)

        # print(f'table = {os.path.basename(initial_data_path)[0:-4]}, count = {np.sum(self.base_histogram, axis=1)}')

    def dump_base_histogram(self, base_dir):
        dir = os.path.join(base_dir, self.table_name)
        FileViewer.detect_and_create_dir(dir)
        for i, data_i in enumerate(self.initial_data):
            initial_data_path = os.path.join(dir, f'data_{i}.npy')
            np.save(initial_data_path, data_i)
        boundary_points_path = os.path.join(dir, 'boundary_points.npy')
        np.save(boundary_points_path, self.boundary_points)
        base_histogram_path = os.path.join(dir, "base_histogram.npy")
        np.save(base_histogram_path, self.base_histogram)
        base_sorted_idxes_path = os.path.join(dir, "base_sorted_idxes.npy")
        np.save(base_sorted_idxes_path, self.base_sorted_idxes)


    def checkpoint_exists(self, base_dir):
        dir = os.path.join(base_dir, self.table_name)
        initial_data_paths = []
        for i in range(self.n_attrs):
            initial_data_path = os.path.join(dir, f'data_{i}.npy')
            initial_data_paths.append(initial_data_path)

        boundary_points_path = os.path.join(dir, 'boundary_points.npy')
        base_histogram_path = os.path.join(dir, "base_histogram.npy")
        base_sorted_idxes_path = os.path.join(dir, "base_sorted_idxes.npy")

        all_paths = []
        all_paths.extend(initial_data_paths)
        all_paths.append(boundary_points_path)
        all_paths.append(base_histogram_path)
        all_paths.append(base_sorted_idxes_path)

        for path in all_paths:
            if not os.path.exists(path):
                return False

        return True


    def load_base_histogram(self, base_dir):
        dir = os.path.join(base_dir, self.table_name)
        initial_data_paths = []
        for i in range(self.n_attrs):
            initial_data_path = os.path.join(dir, f'data_{i}.npy')
            initial_data_paths.append(initial_data_path)
            assert os.path.exists(initial_data_path)

        boundary_points_path = os.path.join(dir, 'boundary_points.npy')
        assert os.path.exists(boundary_points_path)
        base_histogram_path = os.path.join(dir, "base_histogram.npy")
        assert os.path.exists(base_histogram_path)
        base_sorted_idxes_path = os.path.join(dir, "base_sorted_idxes.npy")
        assert os.path.exists(base_sorted_idxes_path)

        self.initial_data = []
        for path in initial_data_paths:
            data_i = np.load(path)
            self.initial_data.append(data_i)
        self.boundary_points = np.load(boundary_points_path)
        self.base_histogram = np.load(base_histogram_path)
        self.base_sorted_idxes = np.load(base_sorted_idxes_path)


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
            else:  # op == '=='
                filter_ranges[i][0] = float(rhs) - 0.5
                filter_ranges[i][1] = float(rhs) + 0.5
            relevant_attrs.add(i)

        for i in relevant_attrs:
            attr_data = self.initial_data[i]
            boundary_points_i = self.boundary_points[i]
            start_idx = np.searchsorted(boundary_points_i, filter_ranges[i][0], side='right')
            start_idx = max(0, start_idx)
            end_idx = np.searchsorted(boundary_points_i, filter_ranges[i][1], side='right')
            end_idx = max(0, end_idx)
            if start_idx >= boundary_points_i.shape[0]:
                histogram[i] = 0
            elif start_idx == end_idx:

                l = 0
                if start_idx > 0:
                    l = self.base_sorted_idxes[i][start_idx - 1]
                r = self.base_sorted_idxes[i][start_idx]
                attr_data = attr_data[l:r]
                idxes = np.searchsorted(attr_data, filter_ranges[i], side='left')
                histogram[i] = 0
                histogram[i][start_idx] = idxes[1] - idxes[0]
            else:
                assert end_idx > start_idx
                left_bin_filter_range = np.array([filter_ranges[i][0], boundary_points_i[start_idx]],
                                                 dtype=filter_ranges.dtype)
                assert left_bin_filter_range[0] <= left_bin_filter_range[1]
                histogram[i][0:start_idx] = 0
                idxes = np.searchsorted(attr_data, left_bin_filter_range, side='left')
                histogram[i][start_idx] = idxes[1] - idxes[0]
                if end_idx < boundary_points_i.shape[0]:
                    right_bin_filter_range = np.array([filter_ranges[i][1], boundary_points_i[end_idx]],
                                                      dtype=filter_ranges.dtype)
                    histogram[i][end_idx:] = 0
                    idxes = np.searchsorted(attr_data, right_bin_filter_range, side='left')
                    histogram[i][end_idx] = idxes[1] - idxes[0]

            if print_flag:
                attr_name = no_attr_map[i]
                print(f'\tattr = {attr_name}')
                print(f'\thistogram = {histogram[i].astype(np.int64)}')
                print(f'\tbase_histogram = {self.base_histogram[i].astype(np.int64)}')
                print(f'\ttable_card = {np.sum(self.base_histogram[i].astype(np.int64))}')
        feature = np.reshape(histogram / self.table_size_threshold, [-1])
        return feature

    # The format of a clause is like 'b.id > 1 and b.date > 100 and b.date <= 1000
    def calc_histogram_feature0(self, filter_conds, print_flag=False):
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
    def __init__(self, tables_info, data_dir, n_bins, keep_checkpoint=False, base_histogram_dir=None):
        # base_histogram_dir = os.path.join(data_dir, 'base_histogram_dir')
        # FileViewer.detect_and_create_dir(base_histogram_dir)
        if keep_checkpoint:
            assert base_histogram_dir is not None
            FileViewer.detect_and_create_dir(base_histogram_dir)
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
        print('\tInitialzing DB states...')
        for i in range(self.n_tables):
            table_name = self.no_table_map[i]
            fname = f'{table_name}.csv'
            path = os.path.join(data_dir, fname)
            th = staticTableHistogram(path,
                                      n_bins=n_bins,
                                      min_vals=attr_ranges_list[i][:, 0],
                                      max_vals=attr_ranges_list[i][:, 1],
                                      attr_no_map=attr_no_map_list[i],
                                      table_size_threshold=table_card_list[i],
                                      base_histogram_dir=base_histogram_dir,
                                      keep_checkpoint=keep_checkpoint)
            self.table_historgrams.append(th)


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

        nlines = len(lines)
        for i in range(nlines):
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

            # if i % 1000 == 0:
            #     print(f'\tBuilding DB states: {(i * 100) // nlines}%', end='\r')
            if i % 10 == 0:
                print(f'\t{i} finished', end='\r')
        db_states = np.array(db_states, dtype=np.float64)
        train_idxes = np.array(train_idxes, dtype=np.int64)
        train_sub_idxes = np.array(train_sub_idxes, dtype=np.int64)
        test_idxes = np.array(test_idxes, dtype=np.int64)
        test_sub_idxes = np.array(test_sub_idxes, dtype=np.int64)
        test_single_idxes = np.array(test_single_idxes, dtype=np.int64)
        return db_states, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes, all_queries
