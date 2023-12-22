import copy
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
sys.path.append("../../")
from . import feature


class dbDataset(Dataset):
    def __init__(self, data):
        self.X = data[0]
        self.Q = data[1]
        self.labels = data[2]
        self.masks= data[3]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Q[idx], self.labels[idx], self.masks[idx]


def create_loaders(train_data, validation_data, test_data, batch_size):
    train_dataset = dbDataset(train_data)
    validation_dataset = dbDataset(validation_data)
    test_dataset = dbDataset(test_data)

    loader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False)

    loader_validation = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False)

    loader_test = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False)

    return loader_train, loader_validation, loader_test #, full_train_data


def normalize_query_features(train_query_features, test_query_features_list, n_possible_joins=None):
    train_std = train_query_features.std(axis=0)

    join_pattern_train_std = None
    if n_possible_joins is not None:
        join_pattern_train_std = train_std[(train_std.shape[0] - n_possible_joins):]

    nonzero_idxes = np.where(train_std > 0)[0]
    # print('-----nonzero_idxes.shape =', nonzero_idxes.shape)
    # print('-----train_std.shape =', train_std.shape)
    # print(nonzero_idxes)
    # print(train_std)
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
        # print('n_possible_joins =', n_possible_joins)
        join_pattern_nonzero_idxes = np.where(join_pattern_train_std > 0)[0]
        join_pattern_dim = join_pattern_nonzero_idxes.shape[0]

        return train_query_features, test_query_features_list, join_pattern_dim
    else:
        return train_query_features, test_query_features_list


def load_random_mask(base_mask, mask_ratio, feature_data_dir):
    if mask_ratio >= 1:
        return None
    path = os.path.join(feature_data_dir, f'random_mask_{mask_ratio}.npy')
    if os.path.exists(path):
        random_mask = np.load(path)
    else:
        n = base_mask.shape[0] * base_mask.shape[1]
        random_mask = np.zeros(shape=[n], dtype=base_mask.dtype)
        m = int(n * mask_ratio)
        random_mask[0:m] = 1
        np.random.shuffle(random_mask)
        random_mask = np.reshape(random_mask, base_mask.shape)
        np.save(path, random_mask)
    return random_mask


def remove_min_max(base_mask):
    n_tasks = (base_mask.shape[1] - 1)
    assert (n_tasks % 4 == 0)
    n_attrs = n_tasks // 4
    random_mask = copy.deepcopy(base_mask)
    for i in range(n_attrs):
        idx = i * 4 + 3
        random_mask[:, idx] = 0
        random_mask[:, idx+1] = 0
    return random_mask


def load_workload_data(cfg):
    wl_type = cfg.dataset.wl_type
    num_parts = cfg.dataset.dynamic_num_parts
    test_from = cfg.dataset.dynamic_test_from
    test_to = cfg.dataset.dynamic_test_to

    db_states, query_featurizations, task_values, task_masks, train_idxes, train_sub_idxes \
        , test_idxes, test_sub_idxes, _, meta_infos = feature.process_workload_data(cfg, wl_type)

    if wl_type == 'static':
        assert train_sub_idxes.shape[0] == 0
        assert test_sub_idxes.shape[0] == 0
    else:  # specifically implemented for the dynamic workload, the first six lines are hard-coded
        num_train_parts = 2 * num_parts + 1
        num_test_parts = 2 * num_parts + 1
        assert test_idxes.shape[0] % num_test_parts == 0
        assert test_sub_idxes.shape[0] % num_test_parts == 0
        num_per_test = test_idxes.shape[0] // num_test_parts
        num_per_test_sub = test_sub_idxes.shape[0] // num_test_parts

        assert train_sub_idxes.shape[0] % num_train_parts == 0
        num_per_train = train_idxes.shape[0] // num_train_parts
        # assert num_per_train == 998 # the STATS dynamic workload
        num_per_train_sub = train_sub_idxes.shape[0] // num_train_parts

        train_start = 0
        train_end = num_per_train * num_parts
        train_sub_start = 0
        train_sub_end = num_per_train_sub * num_parts

        train_idxes = train_idxes[train_start:train_end]
        train_sub_idxes = train_sub_idxes[train_sub_start:train_sub_end]
        train_idxes = np.concatenate([train_idxes, train_sub_idxes])

        train_idxes = np.sort(train_idxes)

        train_db_states = db_states[train_idxes]
        train_query_featurizations = query_featurizations[train_idxes]
        train_task_values = task_values[train_idxes]
        train_task_masks = task_masks[train_idxes]

        test_start = (num_parts + test_from) * num_per_test
        test_end = (num_parts + test_to) * num_per_test
        test_sub_start = (num_parts + test_from) * num_per_test_sub
        test_sub_end = (num_parts + test_to) * num_per_test_sub

        test_sub_idxes = test_sub_idxes[test_sub_start:test_sub_end]

        test_idxes = test_idxes[test_start:test_end]
        test_idxes = np.concatenate([test_idxes, test_sub_idxes])
        test_idxes = np.sort(test_idxes)

        test_db_states = db_states[test_idxes]
        test_query_featurizations = query_featurizations[test_idxes]
        test_task_values = task_values[test_idxes]
        test_task_masks = task_masks[test_idxes]

        n_train = train_idxes.shape[0]
        train_idxes = np.arange(0, n_train, dtype=np.int64)
        test_idxes = np.arange(0, test_idxes.shape[0], dtype=np.int64) + n_train

        db_states = np.concatenate([train_db_states, test_db_states], axis=0, dtype=train_db_states.dtype)
        query_featurizations = np.concatenate([train_query_featurizations, test_query_featurizations], axis=0,
                                              dtype=train_query_featurizations.dtype)
        task_values = np.concatenate([train_task_values, test_task_values], axis=0, dtype=train_task_values.dtype)
        task_masks = np.concatenate([train_task_masks, test_task_masks], axis=0, dtype=train_task_masks.dtype)

    # print('db_states.shape =', db_states.shape)
    # print('query_featurizations.shape =', query_featurizations.shape)
    # print('train_idxes.shape =', train_idxes.shape)
    # print('train_sub_idxes.shape =', train_sub_idxes.shape)
    # print('test_idxes.shape =', test_idxes.shape)
    # print('test_sub_idxes.shape =', test_sub_idxes.shape)

    [n_bins, num_attrs, n_possible_joins, n_tasks] = meta_infos

    train_db_states = db_states[train_idxes]
    train_query_featurizations = query_featurizations[train_idxes]
    train_task_values = task_values[train_idxes]
    train_task_masks = task_masks[train_idxes]

    test_db_states = db_states[test_idxes]
    test_query_featurizations = query_featurizations[test_idxes]
    test_task_values = task_values[test_idxes]
    test_task_masks = task_masks[test_idxes]

    (train_db_states, test_db_states) = normalizations(
        (train_db_states, test_db_states)
    )
    n_bins = train_db_states.shape[1]

    train_query_featurizations, test_query_featurizations_list, join_pattern_dim = normalize_query_features(train_query_featurizations, [test_query_featurizations], n_possible_joins)
    test_query_featurizations = test_query_featurizations_list[0]

    # print(f'num_train = {train_db_states.shape[0]}')
    assert train_db_states.shape[0] == train_query_featurizations.shape[0]
    # print(f'num_test = {test_db_states.shape[0]}')
    assert test_db_states.shape[0] == test_query_featurizations.shape[0]

    # print(f'db_states.shape = {train_db_states.shape}')
    # print(f'query_featurizations.shape = {train_query_featurizations.shape}')

    meta_infos = [n_bins, train_query_featurizations.shape[1], num_attrs, join_pattern_dim, n_tasks]
    meta_infos = tuple(meta_infos)

    return (train_db_states, train_query_featurizations, train_task_values, train_task_masks),\
           (test_db_states, test_query_featurizations, test_task_values, test_task_masks), meta_infos


def _normalize(data, X_mean, X_std, nonzero_idxes):
    norm_data = (data - X_mean)
    norm_data[:, nonzero_idxes] /= X_std[nonzero_idxes]
    return norm_data


def normalizations(datas):
    X = datas[0]
    X_std = X.std(axis=0)
    nonzero_idxes = np.where(X_std > 0)[0]
    X_mean = X.mean(axis=0)
    norm_data = tuple(_normalize(data, X_mean, X_std, nonzero_idxes) for data in datas)
    return norm_data


def get_valid_data(data):
    (db_states, query_featurizations, task_values, task_masks) = data
    s = np.sum(task_masks, axis=1)
    
    idxes = np.where(s != 0)[0]
    db_states = db_states[idxes]
    query_featurizations = query_featurizations[idxes]
    task_values = task_values[idxes]
    task_masks = task_masks[idxes]
    return (db_states, query_featurizations, task_values, task_masks)


# # TODO: futher optimization
def get_task_value_norm_params(train_task_values, train_task_masks, test_task_values, test_task_masks, threshold, buffer_path):
    # to fully simulate a real-world setting
    # the mininum and maximum values of an attribute are based
    # on the valid data according to the task_masks

    if os.path.exists(buffer_path):
        buffer = np.load(buffer_path)
        _threshold = buffer[-1,0]
        assert _threshold == threshold # cfg.encoder.value_range_threshold
        return buffer[0:-1]

    num_tasks = train_task_values.shape[1]
    t_values_trans = train_task_values.T
    t_masks_trans = train_task_masks.T

    test_t_values_trans = test_task_values.T
    test_t_masks_trans = test_task_masks.T

    # an additional array to store the mean, std...
    buffer = np.zeros((num_tasks + 1, 4)).astype(float)
    buffer[-1, 0] = threshold

    for i in range(num_tasks):
        nonzero_idxes = np.where(t_masks_trans[i] > 0.5)
        valid_task_values = t_values_trans[i][nonzero_idxes]
        # bias = valid_task_values.min()
        # max_val = valid_task_values.max()

        test_nonzero_idxes = np.where(test_t_masks_trans[i] > 0.5)
        test_valid_task_values = test_t_values_trans[i][test_nonzero_idxes]

        bias = min(valid_task_values.min(), test_valid_task_values.min())
        max_val = max(valid_task_values.max(), test_valid_task_values.max())

        buffer[i, 1] = bias
        buffer[i, 2] = max_val

        if (max_val - bias) > threshold:
            buffer[i, 0] = 0
            # # make the smallest value equals to 1
            # valid_task_values = valid_task_values - bias + 1
            #
            # # take the log
            # valid_task_values = np.log(valid_task_values)

            log_max_val = math.log(max_val - bias + 1)
            buffer[i, 3] = log_max_val
        else:
            buffer[i, 0] = 1

    np.save(buffer_path, buffer)

    return buffer[0:-1]

def process_task_values(_task_values, task_masks, buffer, original_masks, make_copy=True):
    if make_copy:
        task_values = copy.deepcopy(_task_values)
    else:
        task_values = _task_values
    num_tasks = task_values.shape[1]
    t_values_trans = task_values.T
    t_masks_trans = task_masks.T
    original_masks_trans = original_masks.T

    for i in range(num_tasks):
        nonzero_idxes = np.where(t_masks_trans[i] > 0.5)
        zero_idxes = np.where(t_masks_trans[i] <= 0.5)
        valid_task_values = t_values_trans[i][nonzero_idxes]
        bias = buffer[i, 1]
        max_val = buffer[i, 2]

        _type = buffer[i, 0]

        flag = True
        if _type < 0.5: # take the log
            log_max_val = buffer[i, 3]
            valid_task_values = valid_task_values - bias + 1
            valid_task_values = np.log(valid_task_values)
            valid_task_values /= log_max_val
            minv = np.min(valid_task_values)
            maxv = np.max(valid_task_values)
            if np.isnan(minv) or np.isnan(maxv):
                print(
                    f'task-{i}, log_max_val = {log_max_val}, bias = {bias}, max_val = {max_val}, minv = {minv}, maxv = {maxv}')
        else:
            if bias == max_val:
                t_masks_trans[i][:] = 0
                original_masks_trans[i][:] = 0
                nonzero_idxes = np.where(t_masks_trans[i] > 0.5)
                zero_idxes = np.where(t_masks_trans[i] <= 0.5)
                assert zero_idxes[0].shape[0] == t_masks_trans[i].shape[0]
                flag = False
            else:
                valid_task_values = (valid_task_values - bias) / (max_val - bias)

        if flag:
            t_values_trans[i][nonzero_idxes] = valid_task_values
            t_values_trans[i][zero_idxes] = 0.5
        else:
            assert nonzero_idxes[0].shape[0] == 0
            t_values_trans[i][:] = 0.5

    task_values = t_values_trans.T
    task_masks = t_masks_trans.T
    original_masks = original_masks_trans.T

    return task_values, task_masks, original_masks

def recover_task_values(_task_values, task_masks, buffer, make_copy=True):
    if make_copy:
        task_values = copy.deepcopy(_task_values)
    else:
        task_values = _task_values
    num_tasks = task_values.shape[1]
    t_values_trans = task_values.T
    t_masks_trans = task_masks.T

    for i in range(num_tasks):
        nonzero_idxes = np.where(t_masks_trans[i] > 0.5)
        valid_task_values = t_values_trans[i][nonzero_idxes]
        bias = buffer[i, 1]
        max_val = buffer[i, 2]

        _type = buffer[i, 0]

        if _type < 0.5:  # take the exp
            log_max_val = buffer[i, 3]
            valid_task_values *= log_max_val
            valid_task_values = np.exp(valid_task_values)
            valid_task_values = valid_task_values - 1 + bias
        else:
            valid_task_values = valid_task_values * (max_val - bias) + bias


        t_values_trans[i][nonzero_idxes] = valid_task_values
        task_values = t_values_trans.T

    return task_values, task_masks



def load_data(cfg):
    print('Loading data...')
    workload_data = load_workload_data(cfg)
    (train_data, test_data, meta_infos) = workload_data

    origin_train_task_masks = train_data[-1]
    origin_test_task_masks = test_data[-1]

    (db_states_dim, query_featurizations_dim, num_attrs, n_possible_joins, n_tasks) = meta_infos

    assert (db_states_dim == num_attrs * cfg.dataset.n_bins)
    cfg.dataset.query_part_feature_dim = query_featurizations_dim
    cfg.dataset.join_pattern_dim = n_possible_joins
    cfg.dataset.num_attrs = num_attrs
    cfg.dataset.num_task = n_tasks

    print('Processing data...')

    train_data = get_valid_data(train_data)
    test_data = get_valid_data(test_data)

    (train_db_states, train_query_featurizations, train_task_values, train_task_masks) = train_data
    (test_db_states, test_query_featurizations, test_task_values, test_task_masks) = test_data

    original_test_task_values = copy.deepcopy(test_task_values)

    buffer_path = os.path.join(cfg.dataset.FEATURE_DATA_DIR, 'task_value_norm.npy')
    task_value_norm_params = get_task_value_norm_params(train_task_values, train_task_masks, test_task_values, test_task_masks,
                                                        threshold=cfg.encoder.value_range_threshold, buffer_path=buffer_path)
    train_task_values, train_task_masks, origin_train_task_masks = process_task_values(train_task_values, train_task_masks, task_value_norm_params, origin_train_task_masks)
    test_task_values, test_task_masks, origin_test_task_masks = process_task_values(test_task_values, test_task_masks, task_value_norm_params, origin_test_task_masks)
    processed_train_masks_path = os.path.join(cfg.dataset.FEATURE_DATA_DIR, 'processed_train_masks.npy')
    processed_test_masks_path = os.path.join(cfg.dataset.FEATURE_DATA_DIR, 'processed_test_masks.npy')
    if cfg.dataset.wl_type != 'static':
        processed_train_masks_path = os.path.join(cfg.dataset.FEATURE_DATA_DIR, f'processed_train_masks_{cfg.dataset.dynamic_test_from}_{cfg.dataset.dynamic_test_to}.npy')
        processed_test_masks_path = os.path.join(cfg.dataset.FEATURE_DATA_DIR, f'processed_test_masks_{cfg.dataset.dynamic_test_from}_{cfg.dataset.dynamic_test_to}.npy')
    if not (os.path.exists(processed_test_masks_path) or os.path.exists(processed_train_masks_path)):
        np.save(processed_train_masks_path, origin_train_task_masks)
        np.save(processed_test_masks_path, origin_test_task_masks)

    if cfg.dataset.mask_min_max:
        train_task_masks = remove_min_max(train_task_masks)
        # random_mask = load_random_mask(train_task_masks, cfg.dataset.mask_ratio, cfg.dataset.FEATURE_DATA_DIR)
        # if random_mask is not None:
        #     train_task_masks = train_task_masks * random_mask
        test_task_masks = remove_min_max(test_task_masks)


    # random_mask = load_random_mask(train_task_masks, cfg.dataset.mask_ratio, cfg.dataset.FEATURE_DATA_DIR)
    # if random_mask is not None:
    #     test_task_masks = test_task_masks * random_mask

    # print(train_buffer)
    # randomly select 10% of train data as validation data
    # set the seed for experiments consistency
    # np.random.seed(11)


    #randomly select 10% of train data as validation data
    N_train = train_db_states.shape[0]
    shuffle_idxes = np.arange(0, N_train, dtype=np.int64)
    np.random.shuffle(shuffle_idxes)
    train_db_states = train_db_states[shuffle_idxes]
    train_query_featurizations = train_query_featurizations[shuffle_idxes]
    train_task_values = train_task_values[shuffle_idxes]
    train_task_masks = train_task_masks[shuffle_idxes]

    '''
    The following code snippet is only used for testing
    overfitting issue that might be caused by PyTorch 
    implementation
    '''
    # split the training data into two parts
    N_train = int(N_train * 0.9)
    validation_db_states = train_db_states[N_train:]
    validation_query_featurizations = train_query_featurizations[N_train:]
    validation_task_values = train_task_values[N_train:]
    validation_task_masks = train_task_masks[N_train:]

    train_db_states = train_db_states[0:N_train]
    train_query_featurizations = train_query_featurizations[0:N_train]
    train_task_values = train_task_values[0:N_train]
    train_task_masks = train_task_masks[0:N_train]

    dtype = np.float32
    if cfg.model.use_float64:
        dtype = np.float64

    train_data = (train_db_states.astype(dtype), train_query_featurizations.astype(dtype), train_task_values.astype(dtype), train_task_masks)
    validation_data = (validation_db_states.astype(dtype), validation_query_featurizations.astype(dtype), validation_task_values.astype(dtype), validation_task_masks)
    test_data = (test_db_states.astype(dtype), test_query_featurizations.astype(dtype), test_task_values.astype(dtype), test_task_masks)
    # validation_data = test_data

    # print('train_task_values.shape =', train_task_values.shape)
    # print('train_masks.shape =', train_task_masks.shape)
    # print('test_task_values.shape =', test_task_values.shape)
    # print('test_masks.shape =', test_task_masks.shape)

    return (train_data, validation_data, test_data, original_test_task_values, task_value_norm_params)

