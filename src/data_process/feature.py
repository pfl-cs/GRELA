import os
import time

import numpy as np
import sys
sys.path.append("../")
from src.data_process import general_query_process, schema
from src.utils import file_utils, FileViewer
from src import config
from src.data_process import staticHistogram
from src.data_process import dynamicHistogram

# now, a query is assumed to contain range selection and equi-join condition
class queryFeature(object):
    def __init__(self, table_no_map, attr_no_map_list, attr_no_types_list, attr_ranges_list, possible_join_attrs, keep_filter_info=False):
        """
        :param attr_ranges_list: list with each element is a n x 2 numpy float matrix.
        denoting the ith table has n attrs. These n attrs' range are represented by this matrix.
        :param possible_join_attrs: N * 4 numpy int matrx with each row looks like [table_no1, table_no1.attr_no, table_no2, table_no2.attr_no]
        """

        self.table_no_map, self.attr_no_map_list, self.attr_no_types_list, self.attr_ranges_list \
            = table_no_map, attr_no_map_list, attr_no_types_list, attr_ranges_list

        self.attr_ranges_all = np.concatenate(attr_ranges_list, axis=0, dtype=np.float64)
        self.attr_types_all = np.concatenate(attr_no_types_list, dtype=np.int64)

        # for feature calc only
        self.attr_lbds = np.zeros_like(self.attr_ranges_all, dtype=self.attr_ranges_all.dtype)
        self.attr_range_measures = np.zeros_like(self.attr_ranges_all, dtype=self.attr_ranges_all.dtype)

        self.attr_lbds[:, 0] = self.attr_ranges_all[:, 0]
        self.attr_lbds[:, 1] = self.attr_ranges_all[:, 0]

        tmp = self.attr_ranges_all[:, 1] - self.attr_ranges_all[:, 0]

        self.attr_range_measures[:, 0] = tmp
        self.attr_range_measures[:, 1] = tmp

        self.attr_lbds = np.reshape(self.attr_lbds, [-1])
        self.attr_range_measures = np.reshape(self.attr_range_measures, [-1])
        # print("attr_range_measures_all.shape =", self.attr_range_measures_all.shape)

        self.n_tables = len(attr_ranges_list)
        self.n_attrs_total = self.attr_types_all.shape[0]

        self.maxn_attrs_single_table = self.attr_ranges_list[0].shape[0]
        for i in range(1, self.n_tables):
            table_i = self.attr_ranges_list[i]
            n_attrs = table_i.shape[0]
            if n_attrs > self.maxn_attrs_single_table:
                self.maxn_attrs_single_table = n_attrs

        # print("n_attrs_total =", self.n_attrs_total)
        join_attrs_trans = possible_join_attrs.transpose()

        t1, t1_attr, t2, t2_attr =  join_attrs_trans[0], join_attrs_trans[1], join_attrs_trans[2], join_attrs_trans[3]
        m1 = t1 * self.maxn_attrs_single_table + t1_attr
        m2 = t2 * self.maxn_attrs_single_table + t2_attr
        M = self.maxn_attrs_single_table * self.n_tables

        join_ids = set()
        equi_relations = {}
        for i in range(m1.shape[0]):
            id_1 = m1[i]
            id_2 = m2[i]
            if id_1 <= id_2:
                join_id = id_1 * M + id_2
                symm_join_id = id_2 * M + id_1
            else:
                join_id = id_2 * M + id_1
                symm_join_id = id_1 * M + id_2
            equi_relations[join_id] = symm_join_id
            join_ids.add(join_id)
        join_ids = list(join_ids)
        join_ids.sort()

        self.join_id_no_map = {}
        for i, join_id in enumerate(join_ids):
            self.join_id_no_map[join_id] = i
            symm_join_id = equi_relations[join_id]
            self.join_id_no_map[symm_join_id] = i

        self.n_possible_joins = len(self.join_id_no_map)
        self.keep_filter_info = keep_filter_info

    # join_id: a number from [0, M * M), where M = self.maxn_attrs_single_table * self.n_tables
    def calc_join_ids(self, join_conds):
        join_conds_trans = np.transpose(join_conds)
        m1 = join_conds_trans[0] * self.maxn_attrs_single_table + join_conds_trans[1]
        m2 = join_conds_trans[2] * self.maxn_attrs_single_table + join_conds_trans[3]
        M = self.maxn_attrs_single_table * self.n_tables
        return M * m1 + m2

    # join_no: a number from [0, self.n_possible_joins)
    def calc_join_nos(self, join_conds):
        join_idxes = self.calc_join_ids(join_conds)
        for i in range(join_idxes.shape[0]):
            join_idxes[i] = self.join_id_no_map[join_idxes[i]]
        return join_idxes

    def encode(self, sql_join_conds, sql_attr_ranges_conds, relevant_tables):
        """
        :param sql_join_conds: shape=[m, self.n_possible_joins]
        :param sql_attr_ranges_conds: shape=[self.n_attrs_total * 2]
        :return:
        """
        feature = np.zeros(self.n_tables + self.n_possible_joins + self.n_attrs_total * 2, dtype=np.float64)

        # encode tables appeared in query
        feature[relevant_tables] = 1

        # encode conjunctive join conds
        if sql_join_conds is not None:
            join_id_idxes = self.calc_join_ids(sql_join_conds)
            join_id_idxes += self.n_tables
            feature[join_id_idxes] = 1

        cursor = self.n_tables + self.n_possible_joins

        # encode conjunctive filter conds
        feature[cursor:cursor + self.n_attrs_total * 2] = ((sql_attr_ranges_conds - self.attr_lbds) / self.attr_range_measures) * 2.0 - 1

        if not self.keep_filter_info:
            return feature[0:self.n_tables + self.n_possible_joins]
        return feature

    #encode_batch_w_appeared_tables(
    def encode_batch(self, sql_join_conds_batch, sql_attr_ranges_conds_batch, relevant_tables_list):
        """
        :param sql_join_conds_batch: list of sql_join_conds
        :param sql_attr_ranges_conds_batch: shape=[batch_size, self.n_attrs_total * 2]
        :return:
        """
        batch_size = len(sql_join_conds_batch)
        features = np.zeros(shape=[batch_size,self.n_tables + self.n_possible_joins + self.n_attrs_total * 2], dtype=np.float64)
        for i in range(batch_size):
            relevant_tables = relevant_tables_list[i]
            # encode tables appeared in query
            features[i][relevant_tables] = 1
            sql_join_conds = sql_join_conds_batch[i]
            if sql_join_conds is not None:
                # encode conjunctive join conds
                join_id_idxes = self.calc_join_nos(sql_join_conds)
                join_id_idxes += self.n_tables
                # print(join_id_idxes)
                features[i][join_id_idxes] = 1
        cursor = self.n_tables + self.n_possible_joins
        # encode conjunctive filter conds
        features[:, cursor:cursor + self.n_attrs_total * 2] = ((sql_attr_ranges_conds_batch - self.attr_lbds) / self.attr_range_measures) * 2.0 - 1
        # if self.remove_table_infos:
        #     return features[:, self.n_tables:]
        if not self.keep_filter_info:
            return features[:, 0:self.n_tables+self.n_possible_joins]
        return features



def normalize_query_features(train_query_features, test_query_features_list, n_possible_joins=None):
    train_std = train_query_features.std(axis=0)
    nonzero_idxes = np.where(train_std > 0)[0]
    train_query_features = train_query_features[:, nonzero_idxes]
    train_mean = train_query_features.mean(axis=0)
    train_std = train_std[nonzero_idxes]
    train_query_features = (train_query_features - train_mean) / train_std

    num = len(test_query_features_list)
    for i in range(num):
        test_query_features = test_query_features_list[i]
        test_query_features = test_query_features[:, nonzero_idxes]
        test_query_features = (test_query_features - train_mean) / train_std
        test_query_features_list[i] = test_query_features

    if n_possible_joins is not None:
        join_pattern_train_std = train_std[0:n_possible_joins]
        join_pattern_nonzero_idxes = np.where(join_pattern_train_std > 0)[0]
        join_pattern_dim = join_pattern_nonzero_idxes.shape[0]

        return train_query_features, test_query_features_list, join_pattern_dim
    else:
        return train_query_features, test_query_features_list


def get_valid_data(db_states, query_featurizations, task_values, task_masks, zero_card_sampling_ratio):
    true_cards = task_values[:, 0]
    nonzero_idxes = np.where(true_cards > 0)[0]
    zero_idxes = np.where(true_cards == 0)[0]
    db_states_0, query_featurizations_0, task_values_0, task_masks_0 = \
        db_states[nonzero_idxes], query_featurizations[nonzero_idxes], task_values[nonzero_idxes], task_masks[
            nonzero_idxes]

    np.random.shuffle(zero_idxes)
    n = int(zero_idxes.shape[0] * zero_card_sampling_ratio)
    zero_idxes = zero_idxes[0:n]
    db_states_1, query_featurizations_1, task_values_1, task_masks_1 = \
        db_states[zero_idxes], query_featurizations[zero_idxes], task_values[zero_idxes], task_masks[zero_idxes]
    db_states = np.concatenate((db_states_0, db_states_1), axis=0)
    query_featurizations = np.concatenate((query_featurizations_0, query_featurizations_1), axis=0)
    task_values = np.concatenate((task_values_0, task_values_1), axis=0)
    task_masks = np.concatenate((task_masks_0, task_masks_1), axis=0)
    return db_states, query_featurizations, task_values, task_masks



def process_workload_data(cfg, wl_type=None):
    if wl_type is None:
        wl_type = cfg.dataset.wl_type
    assert cfg.dataset.wl_type == 'static' or cfg.dataset.wl_type == 'ins_heavy'
    if wl_type == 'static':
        if_static_workload = True
    else:
        if_static_workload = False
    workload_dir, feature_data_dir, data_feat_ckpt_dir = config.get_feature_data_dir(cfg, wl_type)
    workload_path = os.path.join(workload_dir, cfg.dataset.mysql_workload_fname)
    FileViewer.detect_and_create_dir(feature_data_dir)

    db_states_path = os.path.join(feature_data_dir, 'db_states.npy')
    query_featurizations_path = os.path.join(feature_data_dir, 'query_featurizations.npy')
    task_values_path = os.path.join(feature_data_dir, 'task_values.npy')
    task_masks_path = os.path.join(feature_data_dir, 'task_masks.npy')
    train_idxes_path = os.path.join(feature_data_dir, 'train_idxes.npy')
    test_idxes_path = os.path.join(feature_data_dir, 'test_idxes.npy')
    meta_infos_path = os.path.join(feature_data_dir, 'meta_infos.npy')
    analytic_functions_path = os.path.join(feature_data_dir, 'analytic_funcs.txt')

    train_sub_idxes_path = os.path.join(feature_data_dir, 'train_sub_idxes.npy')
    test_sub_idxes_path = os.path.join(feature_data_dir, 'test_sub_idxes.npy')
    test_single_idxes_path = os.path.join(feature_data_dir, 'test_single_idxes.npy')

    required_paths = [db_states_path, query_featurizations_path, task_values_path, task_masks_path,
                      train_idxes_path, train_sub_idxes_path, test_idxes_path, test_sub_idxes_path,
                      test_single_idxes_path, meta_infos_path, analytic_functions_path]


    all_files_exist = True
    for path in required_paths:
        if os.path.exists(path) == False:
            all_files_exist = False
            break

    print(f'feature_data_dir = {feature_data_dir}')
    print(f'all_files_exist = {all_files_exist}')
    if all_files_exist:
        db_states = np.load(db_states_path)
        query_featurizations = np.load(query_featurizations_path)
        task_values = np.load(task_values_path)
        task_masks = np.load(task_masks_path)

        train_idxes = np.load(train_idxes_path)
        train_sub_idxes = np.load(train_sub_idxes_path)
        test_idxes = np.load(test_idxes_path)
        test_sub_idxes = np.load(test_sub_idxes_path)
        test_single_idxes = np.load(test_single_idxes_path)
        meta_infos = np.load(meta_infos_path).tolist()
        [db_states_dim, num_attrs, n_possible_joins, n_tasks] = meta_infos

        print('db_states.shape =', db_states.shape)
        print('query_featurizations.shape =', query_featurizations.shape)

        print('train_idxes.shape =', train_idxes.shape)
        print('test_idxes.shape =', test_idxes.shape)

        return (db_states, query_featurizations, task_values, task_masks, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes, meta_infos)

    tables_info = schema.get_tables_info(cfg)
    table_no_map, no_table_map, table_card_list, attr_no_map_list \
        , attr_no_types_list, attr_ranges_list = tables_info

    num_attrs = 0
    for attr_no_map in attr_no_map_list:
        num_attrs += len(attr_no_map)

    keep_filter_info = True
    start = time.time_ns()
    if if_static_workload:
        data_dir = cfg.dataset.data_dir
        data_feat = staticHistogram.staticDBHistogram(tables_info, data_dir, cfg.dataset.n_bins, data_feat_ckpt_dir)
    else:
        data_feat = dynamicHistogram.databaseHistogram(tables_info, workload_path, cfg.dataset.n_bins, data_feat_ckpt_dir)
    db_states, train_idxes, train_sub_idxes, test_idxes, test_sub_idxes, test_single_idxes, all_queries = data_feat.build_db_states(workload_path)

    queries_info = general_query_process.parse_queries(
        all_queries,
        table_no_map,
        attr_no_map_list,
        attr_no_types_list,
        attr_ranges_list,
        delim="||"
    )

    possible_join_attrs, join_conds_list, attr_range_conds_list, relevant_tables_list, \
    filter_conds_list, analytic_functions_list, valid_task_values_list = queries_info

    qF = queryFeature(table_no_map, attr_no_map_list, attr_no_types_list, attr_ranges_list, possible_join_attrs, keep_filter_info=keep_filter_info)
    attr_range_conds_batch = np.array(attr_range_conds_list, dtype=np.float64)
    query_featurizations = qF.encode_batch(join_conds_list, attr_range_conds_batch, relevant_tables_list)

    n_possible_joins = qF.n_possible_joins

    print('query_featurizations.shape =', query_featurizations.shape)
    all_features = np.concatenate([db_states, query_featurizations], axis=1, dtype=db_states.dtype)
    print('all_features.shape =', all_features.shape)

    db_states_dim = db_states.shape[1]
    assert db_states_dim == num_attrs * cfg.dataset.n_bins
    query_featurizations_dim = query_featurizations.shape[1]

    # number analytic functions
    all_analytic_functions = ['count(*)']
    n_tables = len(no_table_map)
    for table_no in range(n_tables):
        table_name = no_table_map[table_no]
        attr_no_map = attr_no_map_list[table_no]
        no_attr_map = {}
        for attr in attr_no_map:
            attr_no = attr_no_map[attr]
            no_attr_map[attr_no] = attr
        n_attrs = len(no_attr_map)

        func_types = ['avg({0:s})', 'sum({0:s})', 'min({0:s})', 'max({0:s})']
        for attr_no in range(n_attrs):
            attr_name = no_attr_map[attr_no]
            table_attr = table_name + '.' + attr_name
            for func_type in func_types:
                all_analytic_functions.append(func_type.format(table_attr))

    analytic_func_task_no_map = {}
    for i, func in enumerate(all_analytic_functions):
        analytic_func_task_no_map[func] = i
    all_analytic_functions = [s + '\n' for s in all_analytic_functions]
    file_utils.write_all_lines(analytic_functions_path, all_analytic_functions)

    n_tasks = len(analytic_func_task_no_map)
    data_size = db_states.shape[0]
    task_masks = np.zeros(shape=[data_size, n_tasks], dtype=np.int64)
    task_values = -np.ones(shape=[data_size, n_tasks], dtype=np.float64)

    for (i, funcs) in enumerate(analytic_functions_list):
        valid_task_values = valid_task_values_list[i]
        if valid_task_values is None:
            continue
        if len(valid_task_values) == 1:
            card = valid_task_values[0]
            if card >= 0:
                idxes = [analytic_func_task_no_map[func] for func in funcs]
                task_values[i][idxes] = valid_task_values
                task_masks[i][0] = 1
            continue
        idxes = [analytic_func_task_no_map[func] for func in funcs]
        task_values[i][idxes] = valid_task_values
        task_masks[i][idxes] = 1

    meta_infos = [db_states_dim, num_attrs, n_possible_joins, n_tasks]
    # histogram_feature_dim, query_part_feature_dim, join_pattern_dim, _n_parts, num_attrs]
    meta_infos = np.array(meta_infos, dtype=np.int64)
    stop = time.time_ns()
    print(f'+++++++total_time = {(stop - start) / 1e6}')


    np.save(db_states_path, db_states)
    np.save(query_featurizations_path, query_featurizations)
    np.save(task_values_path, task_values)
    np.save(task_masks_path, task_masks)
    np.save(train_idxes_path, train_idxes)
    np.save(train_sub_idxes_path, train_sub_idxes)
    np.save(test_idxes_path, test_idxes)
    np.save(test_sub_idxes_path, test_sub_idxes)
    np.save(test_single_idxes_path, test_single_idxes)
    np.save(meta_infos_path, meta_infos)

    print('all_features.shape =', all_features.shape)
    print('db_states.shape =', db_states.shape)
    print('query_featurizations.shape =', query_featurizations.shape)
    print('train_idxes.shape =', train_idxes.shape)
    print('train_sub_idxes.shape =', train_sub_idxes.shape)
    print('test_idxes.shape =', test_idxes.shape)
    print('test_sub_idxes.shape =', test_sub_idxes.shape)
    print('test_single_idxes.shape =', test_single_idxes.shape)

    return (db_states, query_featurizations, task_values, task_masks, train_idxes, train_sub_idxes,
            test_idxes, test_sub_idxes, test_single_idxes, meta_infos.tolist())


if __name__ == '__main__':
    cfg = config.getConfigs()
    process_workload_data(cfg)

# python data_process/feature.py --data STATS --wl_type static
# python data_process/feature.py --data STATS --wl_type ins_heavy
