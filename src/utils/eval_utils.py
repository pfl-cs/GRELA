import numpy as np
import math


def process_error_val(val):
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

def visualize_err(preds, labels):
    assert preds.shape == labels.shape
    abs_labels = np.abs(labels)
    valid_idxes = np.where(abs_labels > 0)[0]
    labels = labels[valid_idxes]
    preds = preds[valid_idxes]

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


def calc_rel_err(preds, labels):
    diff = np.abs(preds - labels)
    rel_err = diff / np.abs(labels)
    return rel_err

