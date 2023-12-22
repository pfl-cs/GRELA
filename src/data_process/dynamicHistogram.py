import os
import numpy as np
import sys
import copy
sys.path.append("../")
from src.utils import file_utils, FileViewer, sql_utils

class tableHistogram(object):
    def __init__(self, initial_data_path, n_bins, min_vals, max_vals, attr_no_map, table_size_threshold):
        self.n_bins = n_bins
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.bin_sizes = (max_vals - min_vals) / self.n_bins
        self.attr_no_map = attr_no_map
        self.n_attrs = len(attr_no_map)
        # self.attr_type_list = attr_type_list
        arange_idxes = np.reshape(np.arange(0, self.n_attrs, dtype=np.int64), [1, -1])
        self.idxes = np.concatenate([arange_idxes, np.zeros(shape=arange_idxes.shape, dtype=arange_idxes.dtype)],
                                    axis=0)
        self.table_size_threshold = table_size_threshold

        if initial_data_path is not None:
            with open(initial_data_path, "r") as reader:
                lines = reader.readlines()
            lines = lines[1:]
            # self.table_size = len(lines)
            # self.histogram = np.zeros(shape=[self.n_attrs, self.n_bins], dtype=np.int64)
            initial_data = []
            for k in range(self.n_attrs):
                initial_data.append([])
            for line in lines:
                terms = line.split(",")
                for i, _term in enumerate(terms):
                    term = _term.strip()
                    if len(term) > 0:
                        # print('term =', term)
                        val = float(term)
                        initial_data[i].append(val)
            initial_data = [np.sort(np.array(data_i, dtype=np.float64)) for data_i in initial_data]
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
            print(f'boundary_points.shape = {boundary_points.shape}, len(lines) = {len(lines)}')

            self.histogram = []
            for i in range(self.n_attrs):
                sorted_idxes = np.searchsorted(initial_data[i], boundary_points[i], side='left')
                print(
                    f'table = {os.path.basename(initial_data_path)[0:-4]}, i = {i}, sorted_idxes[0:10] = {sorted_idxes[0:10]}, sorted_idxes[-10:] = {sorted_idxes[-10:]}')
                assert sorted_idxes[-1] == initial_data[i].shape[0]
                sorted_idxes_left_translation = copy.deepcopy(sorted_idxes)
                sorted_idxes_left_translation[1:] = sorted_idxes_left_translation[0:-1]
                sorted_idxes_left_translation[0] = 0
                diff = sorted_idxes - sorted_idxes_left_translation
                self.histogram.append(np.reshape(diff, [1, -1]))
            self.histogram = np.concatenate(self.histogram, axis=0, dtype=np.float64)
        else:
            self.histogram = np.zeros(shape=[self.n_attrs, self.n_bins], dtype=np.float64)

        # print('histogram.shape =', self.histogram.shape)

    def insert(self, insert_values_str):
        _values = insert_values_str.split(",")
        # values = copy.deepcopy(_values)
        assert len(_values) == self.n_attrs

        values = np.array([float(x) for x in _values], dtype=np.float64)
        self.insert_one_row(values)


    def delete(self, delete_values_str):
        _values = delete_values_str.split(",")
        assert len(_values) == self.n_attrs
        values = np.array([float(x) for x in _values], dtype=np.float64)
        self.delete_one_row(values)

    def update(self, update_values_str):
        terms = update_values_str.split('#')
        assert len(terms) == 2
        _values_0 = terms[0].split(",")
        assert len(_values_0) == self.n_attrs
        values_0 = np.array([float(x) for x in _values_0], dtype=np.float64)
        _values_1 = terms[1].split(",")
        assert len(_values_1) == self.n_attrs
        values_1 = np.array([float(x) for x in _values_1], dtype=np.float64)
        self.delete_one_row(values_0)
        self.insert_one_row(values_1)

    def insert_one_row(self, values):
        # assert values.shape[0] == self.n_attrs
        self.idxes[1] = (values - self.min_vals) // self.bin_sizes
        self.histogram[self.idxes[0], self.idxes[1]] += 1

    def delete_one_row(self, values):
        self.idxes[1] = (values - self.min_vals) // self.bin_sizes
        self.histogram[self.idxes[0], self.idxes[1]] -= 1

    def histogram_feature(self):
        feature = np.reshape(self.histogram / self.table_size_threshold, [-1])
        return feature


class databaseHistogram(object):
    def __init__(self, tables_info, workload_path, n_bins):
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
            th = tableHistogram(None,
                                n_bins=n_bins,
                                min_vals=attr_ranges_list[i][:, 0],
                                max_vals=attr_ranges_list[i][:, 1],
                                attr_no_map=attr_no_map_list[i],
                                table_size_threshold=table_card_list[i])
            self.table_historgrams.append(th)

        lines = file_utils.read_all_lines(workload_path)
        table_path_map = {}
        for line in lines:
            if line.startswith('--'):
                break
            terms = line.split(' ')
            path = terms[4][1:-1]
            if not os.path.exists(path):
                path = path.replace('/home/admin/lpf_files', '/home/lpf')
            assert os.path.exists(path)
            fname = os.path.basename(path)
            table_info = fname[0:-4]
            terms = table_info.split('-')
            table_name = terms[0]
            table_path_map[table_name] = path

        if len(table_path_map) != self.n_tables:
            print('table_path_map =', table_path_map)
            print('self.n_tables =', self.n_tables)
        assert len(table_path_map) == self.n_tables
        for table_name in table_path_map:
            assert table_name in table_no_map

        self.table_historgrams.clear()
        for i in range(self.n_tables):
            table_name = self.no_table_map[i]
            assert table_name in table_path_map
            initial_data_path = table_path_map[table_name]
            th = tableHistogram(initial_data_path,
                                n_bins=n_bins,
                                min_vals=attr_ranges_list[i][:, 0],
                                max_vals=attr_ranges_list[i][:, 1],
                                attr_no_map=attr_no_map_list[i],
                                table_size_threshold=table_card_list[i])
            self.table_historgrams.append(th)

    def process_sql(self, _sql):
        try:
            sql = _sql.lower().strip()
            # if sql.startswith('insert'):
            if sql.startswith('1|'): # insert
                terms = sql.split('|')
                table = terms[1].strip()
                table_no = self.table_no_map[table]
                insert_values_str = terms[2].strip()
                self.table_historgrams[table_no].insert(insert_values_str)
                # print('table =', table)

                # self.table_historgrams[table_no].insert(sql)
            # elif sql.startswith('delete'):
            elif sql.startswith('2|'): #delete
                terms = sql.split('|')
                table = terms[1].strip()
                table_no = self.table_no_map[table]
                delete_values_str = terms[3].strip()
                self.table_historgrams[table_no].delete(delete_values_str)
            elif sql.startswith('update'):
                terms = sql.split('|')
                table = terms[1].strip()
                table_no = self.table_no_map[table]
                update_values_str = terms[3].strip()
                self.table_historgrams[table_no].update(update_values_str)
        except:
            print('sql =', _sql)
            raise Exception()

    def batch_process(self, sqls):
        for sql in sqls:
            self.process_sql(sql)

    def current_histogram_feature(self):
        feature_list = [self.table_historgrams[i].histogram_feature() for i in range(self.n_tables)]
        feature = np.concatenate(feature_list)
        return feature

    def _build_db_states(self, workload_results_path):
        lines = file_utils.read_all_lines(workload_results_path)

        start_line_no = 0
        for i, line in enumerate(lines):
            if line.startswith('--'):
                start_line_no = i + 1
                break

        self.query_and_results = []
        self.split_idxes = []
        j = 0
        num_inserts = 0
        k = start_line_no + 1

        batch_sqls = []

        self.histogram_features = []
        self.num_inserts_before_queries = []
        self.train_idxes = []
        self.train_sub_idxes = []
        self.test_idxes = []
        self.test_sub_idxes = []
        self.test_single_idxes = []

        if k >= len(lines):
            return
        nlines = len(lines)

        for i, _line in enumerate(lines):
            if i < k:
                continue
            line = _line.strip().lower()
            if line.startswith("insert") or line.startswith("delete") or line.startswith("update"):
                batch_sqls.append(line)
                num_inserts += 1
                # DH.add(line)
            elif line.startswith("t"): # train or train_sub or test or test_sub
                if i >= start_line_no:
                    terms = line.split("||")
                    n_terms = len(terms)
                    assert n_terms == 1 or n_terms == 3

                    task_values_str = "-1"
                    if n_terms == 3:
                        task_values_str = terms[1]
                        if task_values_str == "-2":
                            task_values_str = "-1"
                    if line.startswith("train"):
                        if line.startswith("train_sub"):
                            self.train_sub_idxes.append(j)
                        else:
                            assert line.startswith("train_query")
                            self.train_idxes.append(j)
                    else: # line.startswith("train"):
                        if line.startswith("test_sub"):
                            self.test_sub_idxes.append(j)
                        elif line.startswith("test_single"):
                            self.test_single_idxes.append(j)
                        else:
                            self.test_idxes.append(j)
                    j += 1
                    query_str = terms[0]
                    idx = query_str.find(':')
                    query_str = query_str[idx + 2:]
                    self.query_and_results.append(query_str + "||" + task_values_str + "\n")
                    self.batch_process(batch_sqls)
                    curr_feature = self.current_histogram_feature()
                    self.histogram_features.append(curr_feature)
                    self.num_inserts_before_queries.append(num_inserts)
                    batch_sqls = []
            elif line.startswith('--'):
                self.split_idxes.append(j)

            if i % 10000 == 0:
                print(f'\tBuilding DB states: {(i * 100) // nlines}%', end='\r')
        self.split_idxes.append(j)


    def build_db_states(self, workload_path):
        self._build_db_states(workload_path)
        histogram_features = np.array(self.histogram_features, dtype=np.float64)
        num_inserts_before_queries = np.array(self.num_inserts_before_queries, dtype=np.int64)
        train_idxes = np.array(self.train_idxes, dtype=np.int64)
        train_sub_idxes = np.array(self.train_sub_idxes, dtype=np.int64)
        test_idxes = np.array(self.test_idxes, dtype=np.int64)
        test_sub_idxes = np.array(self.test_sub_idxes, dtype=np.int64)
        test_single_idxes = np.array(self.test_single_idxes, dtype=np.int64)
        return histogram_features, train_idxes, train_sub_idxes,\
               test_idxes, test_sub_idxes, test_single_idxes, self.query_and_results
        # return self.query_and_results, self.split_idxes, histogram_features, num_inserts_before_queries, \
        #        train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes


