import numpy as np
import math


def process_error_val(val):
    if val is None:
        return 'None'
    if val < 100:
        s = str(round(val, 2))
        terms = s.split('.')
        if len(terms) == 1:
            s += '.00'
        elif len(terms[1]) == 1:
            s += '0'

        return f'${s}$'

    if val >= 1e10:
        return '>$10^{10}$'
    if val < 1e5:
        int_val = int(val)
        s = format(int_val, ',d')
        return f'${s}$'
    exponent = int(math.log10(val))
    x = math.pow(10, exponent)
    a = val / x

    a_str = str(round(a,1))
    terms = a_str.split('.')
    if len(terms) == 1:
        a_str += '.0'
    return f'${a_str}$$\\cdot$$10^{exponent}$'


def process_error_val0(val):
    if val < 100:
        s = str(round(val, 2))
        terms = s.split('.')
        if len(terms) == 1:
            s += '.00'
        elif len(terms[1]) == 1:
            s += '0'

        return f'${s}$'

    if val >= 1e10:
        return '>$10^{10}$'
    if val < 1e5:
        int_val = int(val)
        s = format(int_val, ',d')
        return f'${s}$'
    exponent = int(math.log10(val))
    x = math.pow(10, exponent)
    a = val / x

    a_str = str(round(a,1))
    terms = a_str.split('.')
    if len(terms) == 1:
        a_str += '.0'
    return f'${a_str}$$\\cdot$$10^{exponent}$'


def process_ratio(_ratio):
    ratio = round(_ratio, 3)
    if ratio < 0.001:
        return '<$0.001$'
    else:
        return f'${round(ratio, 3)}$'

def visualize_err0(preds, labels):
    print(f'preds.shape =', preds.shape)
    assert preds.shape == labels.shape
    abs_labels = np.abs(labels)
    valid_idxes = np.where(abs_labels > 0)[0]
    labels = labels[valid_idxes]
    preds = preds[valid_idxes]
    print(f'after processing, preds.shape =', preds.shape)

    err = calc_rel_err(preds, labels)
    err = np.sort(err)
    recall_vals = [0.5, 1, 5, 10]
    recalls = np.array(recall_vals, dtype=np.float64)


    n_vals = err.shape[0]
    idxes = np.searchsorted(err, recalls, side='left')
    idxes = idxes.tolist()
    results = []
    results_wo_dollars = []
    for idx in idxes:
        result = process_ratio(1.0 * idx / n_vals)
        results.append(result)
        results_wo_dollars.append(result[1:-1])
    # result_str = ' & '.join(results)
    # result_wo_dollars_str = ', '.join(results_wo_dollars)
    recall_tag_str_list = [f'Recall-{x}' for x in recall_vals]
    max_len = len(recall_tag_str_list[0]) + 1
    n_recalls = len(recall_vals)
    for i in range(n_recalls):
        s = recall_tag_str_list[i]
        tmp = ' ' * (max_len - len(s))
        recall_tag_str_list[i] = s + tmp
    # print(result_str)
    recall_tags = '|'.join(recall_tag_str_list)
    l = (len(recall_tags) - 8) // 2
    # s = '-' * l
    # print(f'{s}RelError{s}')
    print(f'    RelError:')
    print(f'\t{recall_tags}')
    # print('-' * len(s))
    for i in range(n_recalls):
        s = results_wo_dollars[i]
        tmp = ' ' * (max_len - len(s))
        results_wo_dollars[i] = s + tmp
    result_wo_dollar_str = '|'.join(results_wo_dollars)
    print(f'\t{result_wo_dollar_str}')


def visualize_err(preds, labels, err_type='rel_error'):
    assert preds.shape == labels.shape
    abs_labels = np.abs(labels)
    valid_idxes = np.where(abs_labels > 0)[0]
    labels = labels[valid_idxes]

    preds = preds[valid_idxes]
    # print('-' * 50)
    # print(f'labels.shape = {labels.shape}')
    # print(f'preds.shape = {preds.shape}')
    # print('-' * 50)

    if err_type == 'recall':
        err = calc_rel_err(preds, labels)
        err = np.sort(err)
        group1 = np.array([0.5, 1, 10, 100], dtype=np.float64)
        group2 = np.array([0.1, 0.5, 1, 5], dtype=np.float64)
        group3 = np.array([0.5, 1, 5, 10], dtype=np.float64)
        # recall_groups = [group1, group2, group3]
        recall_groups = [group3]

        s = '-' * 10
        print(f'{s}{err_type}{s}')
        n_vals = err.shape[0]
        for recalls in recall_groups:
            idxes = np.searchsorted(err, recalls, side='left')
            idxes = idxes.tolist()
            results = []
            results_wo_dollars = []
            for idx in idxes:
                result = process_ratio(1.0 * idx / n_vals)
                results.append(result)
                results_wo_dollars.append(result[1:-1])
            result_str = ' & '.join(results)
            result_wo_dollars_str = ', '.join(results_wo_dollars)
            print(result_str)
            print(result_wo_dollars_str)
    elif err_type == 'rel_error':
        err = calc_rel_err_w_bound(preds, labels)
        err = np.sort(err).tolist()
        n = preds.shape[0]
        num_padding = n - len(err)
        padding_list = [None] * num_padding
        err.extend(padding_list)
        print(f'len(error) = {len(err)}')
        # ratios = [0.5, 0.9, 0.95, 0.99]
        ratios = [0.5, 0.75, 0.9, 0.95]

        error_vals = []
        for ratio in ratios:
            idx = int(n * ratio)
            # print(f'idx = {idx}')
            error_vals.append(err[idx])

        results = []
        for val in error_vals:
            results.append(process_error_val(val))
        # results.append(str(mean_err))
        result_str = ' & '.join(results)
        print(result_str)
    else:
        err = generic_calc_q_error(preds, labels)
        err = np.sort(err)
        mean_err = np.mean(err)
        n = err.shape[0]
        print(f'q_error.shape = {err.shape}')
        # ratios = [0.5, 0.9, 0.95, 0.99]
        ratios = [0.5, 0.75, 0.9, 0.95]

        error_vals = []
        for ratio in ratios:
            idx = int(n * ratio)
            # print(f'idx = {idx}')
            error_vals.append(err[idx])

        results = []
        for val in error_vals:
            results.append(process_error_val(val))
        results.append(str(mean_err))
        result_str = ' & '.join(results)
        print(result_str)

def visualize_err0(preds, labels, err_type='rel_error'):
    assert preds.shape == labels.shape
    abs_labels = np.abs(labels)
    valid_idxes = np.where(abs_labels > 0)[0]
    labels = labels[valid_idxes]

    preds = preds[valid_idxes]
    # print('-' * 50)
    # print(f'labels.shape = {labels.shape}')
    # print(f'preds.shape = {preds.shape}')
    # print('-' * 50)

    if err_type == 'recall':
        err = calc_rel_err(preds, labels)
        err = np.sort(err)
        group1 = np.array([0.5, 1, 10, 100], dtype=np.float64)
        group2 = np.array([0.1, 0.5, 1, 5], dtype=np.float64)
        group3 = np.array([0.5, 1, 5, 10], dtype=np.float64)
        # recall_groups = [group1, group2, group3]
        recall_groups = [group3]

        s = '-' * 10
        print(f'{s}{err_type}{s}')
        n_vals = err.shape[0]
        for recalls in recall_groups:
            idxes = np.searchsorted(err, recalls, side='left')
            idxes = idxes.tolist()
            results = []
            results_wo_dollars = []
            for idx in idxes:
                result = process_ratio(1.0 * idx / n_vals)
                results.append(result)
                results_wo_dollars.append(result[1:-1])
            result_str = ' & '.join(results)
            result_wo_dollars_str = ', '.join(results_wo_dollars)
            print(result_str)
            print(result_wo_dollars_str)
    elif err_type == 'rel_error':
        err = calc_rel_err_w_bound(preds, labels)
        err = np.sort(err).tolist()
        n = preds.shape[0]
        num_padding = n - len(err)
        padding_list = [None] * num_padding
        err.extend(padding_list)
        print(f'len(error) = {len(err)}')
        # ratios = [0.5, 0.9, 0.95, 0.99]
        ratios = [0.5, 0.75, 0.9, 0.95]

        error_vals = []
        for ratio in ratios:
            idx = int(n * ratio)
            # print(f'idx = {idx}')
            error_vals.append(err[idx])

        results = []
        for val in error_vals:
            results.append(process_error_val(val))
        # results.append(str(mean_err))
        result_str = ' & '.join(results)
        print(result_str)
    else:
        err = generic_calc_q_error(preds, labels)
        err = np.sort(err)
        mean_err = np.mean(err)
        n = err.shape[0]
        print(f'q_error.shape = {err.shape}')
        # ratios = [0.5, 0.9, 0.95, 0.99]
        ratios = [0.5, 0.75, 0.9, 0.95]

        error_vals = []
        for ratio in ratios:
            idx = int(n * ratio)
            # print(f'idx = {idx}')
            error_vals.append(err[idx])

        results = []
        for val in error_vals:
            results.append(process_error_val(val))
        results.append(str(mean_err))
        result_str = ' & '.join(results)
        print(result_str)


def calc_rel_err(preds, labels):
    diff = np.abs(preds - labels)
    rel_err = diff / np.abs(labels)
    return rel_err

def calc_rel_err_w_bound(preds, labels, bound=1e50):
    assert preds.shape == labels.shape
    valid_idxes = np.where(preds < bound)
    num = preds.shape[0]
    preds = preds[valid_idxes]
    labels = labels[valid_idxes]
    diff = np.abs(preds - labels)
    rel_err = diff / np.abs(labels)
    return rel_err

def generic_calc_q_error(card_preds, true_cards, if_print=False):
    true_cards += 1e-8
    card_preds = np.clip(card_preds, a_min=1e-8, a_max=None)
    q_error_1 = card_preds / true_cards
    q_error_2 = true_cards / card_preds
    q_error = np.array([q_error_1, q_error_2], dtype=np.float64)
    q_error = np.max(q_error, axis=0)
    q_error = np.reshape(q_error, [-1])
    if if_print == True:
        diff = card_preds - true_cards
        idxes = np.where(diff < 0)[0]
        print('idxes.shape =', idxes.shape, 'true_cards.shape =', true_cards.shape)
    return q_error

def generic_lower_q_error(card_preds, true_cards):
    true_cards += 1e-8
    card_preds = np.clip(card_preds, a_min=1e-8, a_max=None)
    q_error = true_cards / card_preds
    return q_error

def generic_upper_q_error(card_preds, true_cards):
    true_cards += 1e-8
    card_preds = np.clip(card_preds, a_min=1e-8, a_max=None)
    q_error = card_preds / true_cards
    return q_error

def query_driven_calc_card_preds(preds, var_preds, join_cards, y_type, card_log_scale):
    card_preds = preds

    if_card_log = False
    if y_type > 0:
        card_preds = preds * join_cards
    else:
        if_card_log = bool(card_log_scale)

    card_preds_2 = None
    if if_card_log == True:
        card_preds = np.exp(preds)
        card_preds_2 = np.exp(preds + var_preds / 2)
    return card_preds, card_preds_2

def query_driven_calc_q_error(preds, var_preds, true_cards, join_cards, y_type, card_log_scale):
    # min_val = np.min(preds)
    # assert min_val > 0
    card_preds, card_preds_2 = query_driven_calc_card_preds(preds, var_preds, join_cards, y_type, card_log_scale)

    q_error_2 = None
    if card_preds_2 is not None:
        q_error_2 = generic_calc_q_error(card_preds_2, true_cards)

    q_error_1 = generic_calc_q_error(card_preds, true_cards)
    return q_error_1, q_error_2

def calc_fpr(preds, labels, masks):
    cards = labels[:,0]
    card_preds = preds[:,0]
    masks = masks[:,0]
    idxes = np.where(masks > 0.5)[0]
    cards = cards[idxes]
    card_preds = card_preds[idxes]
    idxes = np.where(cards == 0)[0]
    zero_cards = cards[idxes]
    card_preds = card_preds[idxes]
    idxes = np.where(card_preds != 0)[0]
    nonzero_card_preds = card_preds[idxes]
    n = zero_cards.shape[0]
    if n == 0:
        return 0
    fpr = nonzero_card_preds.shape[0] / n
    return fpr
