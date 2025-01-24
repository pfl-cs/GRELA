import os
import argparse
from distutils.util import strtobool
from yacs.config import CfgNode as CN
import math
from pathlib import Path

def set_cfg(cfg):
    if cfg is None:
        return cfg

    # ----------------------------------------------------------------------- #
    # Dataset options
    # ----------------------------------------------------------------------- #
    cfg.dataset = CN()

    # Name of the dataset
    cfg.dataset.name = 'STATS'
    cfg.dataset.wl_type = 'static'
    # TODO: Params that need to be overwrritten
    cfg.dataset.mask_ratio = 1
    cfg.dataset.project_root = '.'
    cfg.dataset.FEATURE_DATA_DIR = ""
    cfg.dataset.mask_min_max = True
    cfg.dataset.workload_fname = 'simplified_workload.sql'
    # cfg.dataset.workload_fname = 'mysql_workload.sql'

    cfg.dataset.tables_info_path = ''
    cfg.dataset.create_tables_path = ''

    # histogram feature params
    cfg.dataset.n_bins = 128
    cfg.dataset.num_attrs = 43

    # query part feature params
    cfg.dataset.query_part_feature_dim = 96
    cfg.dataset.join_pattern_dim = 11
    cfg.dataset.num_task = -1

    cfg.dataset.dynamic_num_parts = 3
    cfg.dataset.dynamic_test_from = 2
    cfg.dataset.dynamic_test_to = 3


    # ----------------------------------------------------------------------- #
    # Running options
    # ----------------------------------------------------------------------- #
    cfg.run = CN()

    # Training (and validation) pipeline mode
    cfg.run.mode = 'standard'
    cfg.run.ckpt_dir = ''
    cfg.run.exp_dir = ''
    cfg.run.encoder_ckpt_fname = 'encoder'

    # TODO: Params that need to be overwrritten
    cfg.run.batch_size = 128  # 16
    cfg.run.gpu = 4
    cfg.run.train_model = True
    cfg.run.force_retrain = False
    cfg.run.eval_model = False
    cfg.run.MAX_CKPT_KEEP_NUM = 1

    cfg.run.pretrain_encoder = False
    cfg.run.freeze_encoder = False
    cfg.run.debug_mode = False


    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    cfg.model = CN()
    cfg.model.use_float64 = False
    cfg.model.dropout_rate = 0
    cfg.model.query_emb_dim = 1024 # 512

    # Loss function: cross_entropy, mse
    cfg.model.loss_fun = 'mse'
    # size average for loss function. 'mean' or 'sum'
    cfg.model.size_average = 'mean'

    # ----------------------------------------------------------------------- #
    # Parameters wrt the Encoder module
    # ----------------------------------------------------------------------- #
    cfg.encoder = CN()
    # this should be udpated everytime
    # this one listed below should not used as default
    cfg.encoder.num_s_self_attn_layers = 3
    cfg.encoder.num_q_cross_attn_layers = 3
    cfg.encoder.num_attn_heads = 8
    cfg.encoder.num_weight_tied_layers = 10
    cfg.encoder.attn_head_key_dim = 1024 # 512,
    cfg.encoder.pooling_mlp_hidden_dim = 1024  # 512
    cfg.encoder.attn_mlp_hidden_dim = 1024  # 512
    cfg.encoder.use_s_ff = True
    cfg.encoder.use_q_ff = True
    cfg.encoder.ff_mlp_num_layers = 3
    cfg.encoder.ff_mlp_hidden_dim = 1024  # 512
    # cfg.encoder.value_range_threshold = 1e3
    cfg.encoder.value_range_threshold = 0


    # ----------------------------------------------------------------------- #
    # Parameters wrt the Graph module
    # ----------------------------------------------------------------------- #
    cfg.graph = CN()
    cfg.graph.num_attn_layers = 3
    cfg.graph.num_attn_heads = 8
    cfg.graph.num_weight_tied_layers = 10
    cfg.graph.attn_mlp_hidden_dim = 1024  # 512
    cfg.graph.use_ff = True
    cfg.graph.ff_mlp_num_layers = 3
    cfg.graph.ff_mlp_hidden_dim = 1024  # 512
    cfg.graph.fix_keys_in_attn = False

    # ----------------------------------------------------------------------- #
    # Optimizer options
    # ----------------------------------------------------------------------- #
    cfg.optim = CN()

    # optimizer: sgd, adam
    cfg.optim.optimizer = 'adam'

    # Base learning rate
    cfg.optim.base_lr = 1e-4 # 0.01

    # L2 regularization
    cfg.optim.weight_decay = 5e-4

    # SGD momentum
    cfg.optim.momentum = 0.9

    # scheduler: none, step, cos
    cfg.optim.scheduler = 'none'
    # cfg.optim.scheduler = 'step'

    # Steps for 'steps' policy (in epochs)
    cfg.optim.steps = [30, 60, 90]

    # Learning rate multiplier for 'steps' policy
    cfg.optim.lr_decay = 0.1

    # Maximal number of epochs
    cfg.optim.max_epoch = 400

    cfg.sampling = CN()
    cfg.sampling.sampling_method = 'unisamp'
    cfg.sampling.sampling_ratio = 0.1



'''
****************************************************************
Get the user-defined arguments from command line directly
****************************************************************
'''
def get_arg_parser():
    parser = argparse.ArgumentParser(description='GRELA')

    # params of cfg.dataset
    parser.add_argument('--data', type=str, default='STATS', help='')
    parser.add_argument('--wl_type', type=str, default='static', help='')
    parser.add_argument('--mask_ratio', type=float, default=1, help='')
    parser.add_argument('--mask_min_max', type=lambda x: bool(strtobool(x)), default=True, help='')
    parser.add_argument('--workload_base_dir', type=str, default='../data/STATS/workload/', help='')
    parser.add_argument('--n_bins', type=int, default=128, help='')

    # params of cfg.train
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--train_model', type=lambda x: bool(strtobool(x)), default=False, help='')
    parser.add_argument('--force_retrain', type=lambda x: bool(strtobool(x)), default=False, help='')
    parser.add_argument('--eval_model', type=lambda x: bool(strtobool(x)), default=False, help='')
    parser.add_argument('--freeze_encoder', type=lambda x: bool(strtobool(x)), default=False, help='')
    parser.add_argument('--gpu', type=int, default=7, help='')
    parser.add_argument('--pretrain_encoder', type=lambda x: bool(strtobool(x)), default=False, help='')
    parser.add_argument('--MAX_CKPT_KEEP_NUM', type=int, default=3, help='')
    parser.add_argument('--task', type=str, default='aqp', help='')
    parser.add_argument('--debug_mode', type=lambda x: bool(strtobool(x)), default=False, help='')

    # params of cfg.model
    parser.add_argument('--query_emb_dim', type=int, default=1024, help='')

    # shared params between cfg.encoder and cfg.graph
    parser.add_argument('--attn_mlp_hidden_dim', type=int, default=1024, help='')
    parser.add_argument('--ff_mlp_hidden_dim', type=int, default=1024, help='')
    parser.add_argument('--ff_mlp_num_layers', type=int, default=3, help='')
    parser.add_argument('--num_weight_tied', type=int, default=10, help='')

    # params of cfg.encoder
    parser.add_argument('--num_s_self_attn_layers', type=int, default=3, help='')
    parser.add_argument('--num_q_cross_attn_layers', type=int, default=3, help='')
    parser.add_argument('--pooling_mlp_hidden_dim', type=int, default=1024, help='')
    parser.add_argument('--attn_head_key_dim', type=int, default=1024, help='')
    parser.add_argument('--use_s_ff', type=lambda x: bool(strtobool(x)), default=True, help='')
    parser.add_argument('--use_q_ff', type=lambda x: bool(strtobool(x)), default=True, help='')

    # params of cfg.graph
    parser.add_argument('--num_g_attn_layers', type=int, default=3, help='')
    parser.add_argument('--use_g_ff', type=lambda x: bool(strtobool(x)), default=True, help='')
    parser.add_argument('--fix_keys_in_attn', type=lambda x: bool(strtobool(x)), default=False, help='')

    # params of cfg.optim
    parser.add_argument('--max_epoch', type=int, default=400, help='')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='')

    # params of cfg.optim
    parser.add_argument('--sampling_method', type=str, default='unisamp',
                        help='')
    parser.add_argument('--sampling_ratio', type=float, default=0.1,
                        help='')



    args = parser.parse_args()
    return args


def overwrite_from_args(args, cfg):
    # overwrite some default config parameters from the arg-parser
    cfg.dataset.name = args.data
    cfg.dataset.wl_type = args.wl_type
    cfg.dataset.mask_ratio = args.mask_ratio
    cfg.dataset.mask_min_max = args.mask_min_max
    cfg.dataset.workload_base_dir = args.workload_base_dir
    cfg.dataset.n_bins = args.n_bins

    # cfg.model
    cfg.model.query_emb_dim = args.query_emb_dim

    # cfg.encoder & cfg.graph
    cfg.encoder.num_weight_tied_layers = args.num_weight_tied
    cfg.graph.num_weight_tied_layers = args.num_weight_tied

    cfg.encoder.attn_mlp_hidden_dim = args.attn_mlp_hidden_dim
    cfg.graph.attn_mlp_hidden_dim = args.attn_mlp_hidden_dim
    
    cfg.encoder.ff_mlp_num_layers = args.ff_mlp_num_layers
    cfg.encoder.ff_mlp_hidden_dim = args.ff_mlp_hidden_dim
    cfg.graph.ff_mlp_num_layers = args.ff_mlp_num_layers
    cfg.graph.ff_mlp_hidden_dim = args.ff_mlp_hidden_dim

    # cfg.encoder
    cfg.encoder.pooling_mlp_hidden_dim = args.pooling_mlp_hidden_dim
    cfg.encoder.num_s_self_attn_layers = args.num_s_self_attn_layers
    cfg.encoder.num_q_cross_attn_layers = args.num_q_cross_attn_layers
    cfg.encoder.attn_head_key_dim = args.attn_head_key_dim

    cfg.encoder.use_s_ff = args.use_s_ff
    cfg.encoder.use_q_ff = args.use_q_ff

    # cfg.graph
    cfg.graph.num_attn_layers = args.num_g_attn_layers
    cfg.graph.use_ff = args.use_g_ff
    cfg.graph.fix_keys_in_attn = args.fix_keys_in_attn


    cfg.run.batch_size = args.batch_size
    cfg.run.train_model = args.train_model
    cfg.run.eval_model = args.eval_model
    cfg.run.force_retrain = args.force_retrain
    cfg.run.freeze_encoder = args.freeze_encoder
    cfg.run.gpu = args.gpu
    cfg.run.pretrain_encoder = args.pretrain_encoder
    cfg.run.MAX_CKPT_KEEP_NUM = args.MAX_CKPT_KEEP_NUM
    cfg.run.task = args.task

    cfg.optim.base_lr = args.base_lr
    cfg.optim.max_epoch = args.max_epoch

    cfg.sampling.sampling_method = args.sampling_method
    cfg.sampling.sampling_ratio = args.sampling_ratio




def set_project_root(cfg, project_root):
    cfg.dataset.project_root = project_root
    workload_dir = os.path.join(project_root, f'data/{cfg.dataset.name}/workload/{cfg.dataset.wl_type}')
    cfg.dataset.FEATURE_DATA_DIR = os.path.join(workload_dir, f'histogram_{cfg.dataset.n_bins}_features')
    cfg.run.ckpt_dir = os.path.join(project_root, f'ckpt/{cfg.dataset.name}/{cfg.dataset.wl_type}')
    cfg.run.exp_dir = os.path.join(project_root, f'exp/{cfg.dataset.name}/{cfg.dataset.wl_type}')


def getConfigs():
    cfg = CN()
    set_cfg(cfg)
    args = get_arg_parser()
    overwrite_from_args(args, cfg)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = Path(cur_dir).parent.absolute()
    set_project_root(cfg, str(project_root))

    # params of cfg.dataset
    cfg.dataset.data_dir = f'../data/{cfg.dataset.name}/data'
    cfg.dataset.tables_info_path = os.path.join(cfg.dataset.data_dir, 'tables_info.txt')
    cfg.dataset.workload_base_dir = f'../data/{cfg.dataset.name}/workload/'
    static_workload_dir = get_workload_dir(cfg, wl_type='static')
    cfg.dataset.create_tables_path = os.path.join(static_workload_dir, 'create_tables.sql')

    return cfg

def get_model_ckpt_fname(cfg, model_name):
    assert model_name is not None
    if model_name is None:
        model_name = cfg.run.encoder_ckpt_fname
    s1 = f'{int(cfg.encoder.use_s_ff)}{int(cfg.encoder.use_q_ff)}{int(cfg.graph.use_ff)}{int(cfg.graph.fix_keys_in_attn)}'
    s2 = f'{cfg.encoder.num_s_self_attn_layers}{cfg.encoder.num_q_cross_attn_layers}{cfg.graph.num_attn_layers}{cfg.encoder.ff_mlp_num_layers}'
    s3 = f'{int(math.log2(cfg.encoder.attn_head_key_dim+1))}_{int(math.log2(cfg.model.query_emb_dim+1))}'
    s4 = f'{int(math.log2(cfg.encoder.pooling_mlp_hidden_dim+1))}_{int(math.log2(cfg.encoder.attn_mlp_hidden_dim+1))}_{int(math.log2(cfg.encoder.ff_mlp_hidden_dim+1))}'
    model_ckpt_fname = f'{model_name}_{s1}_{s2}_{s3}_{s4}'
    return model_ckpt_fname


def get_workload_dir(cfg, wl_type=None):
    if wl_type is None:
        wl_type = cfg.dataset.wl_type
    workload_dir = os.path.join(cfg.dataset.workload_base_dir, wl_type)
    return workload_dir

def get_feature_data_dir(cfg, wl_type=None):
    workload_dir = get_workload_dir(cfg, wl_type)
    data_featurizations_type = f'histogram_{cfg.dataset.n_bins}'
    feature_data_dir = os.path.join(workload_dir, f'{data_featurizations_type}_features')

    return workload_dir, feature_data_dir


def get_rel_error_dir(cfg, wl_type=None):
    if wl_type is None:
        wl_type = cfg.dataset.wl_type

    if wl_type != 'static' and wl_type != 'group_by' and wl_type != 'enrich':
        rel_error_dir = os.path.join(cfg.run.exp_dir, f'{wl_type}_after_{cfg.dataset.dynamic_test_from}/rel_error')
    elif wl_type == 'group_by':
        rel_error_dir = os.path.join(cfg.run.exp_dir, 'group_by/rel_error')
    elif wl_type == 'enrich':
        rel_error_dir = os.path.join(cfg.run.exp_dir, 'enrich/rel_error')
    else:
        rel_error_dir = os.path.join(cfg.run.exp_dir, 'static/rel_error')

    return rel_error_dir
