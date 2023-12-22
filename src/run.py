import sys
sys.path.append("../")
from data_process.loader import load_data, create_loaders, recover_task_values
from utils import ckpt_utils, FileViewer, eval_utils, file_utils
from utils.train_utils import compute_loss, create_optimizer, create_scheduler
from model.GRELA import GRELA
from config import getConfigs, get_model_ckpt_fname
import torch
import os
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def get_valid_labels(labels, preds, masks):
    labels_list = torch.reshape(labels, (-1,))
    preds_list = torch.reshape(preds, (-1,))
    masks_list = torch.reshape(masks, (-1,))
    idxs = masks_list > 0.5

    return labels_list[idxs], preds_list[idxs]


def train_model(cfg, device, loaders, model, optimizer, scheduler, max_epoch, start_epoch=0, best_loss=1e50):
    if not os.path.exists(cfg.run.ckpt_dir):
        print(f"[INFO] creating the directory {cfg.run.ckpt_dir} to save checkpoints")
        os.makedirs(cfg.run.ckpt_dir)

    model.to(device)
    model.train()
    loader_train = loaders[0]
    loader_val = loaders[1]

    for cur_epoch in range(0, max_epoch):
        train_loss = train_model_epoch(device, loader_train, model, optimizer, scheduler)
        train_loss = round(float(train_loss), 5)
        val_loss, _ = eval_model(device, loader_val, model)
        val_loss = round(float(val_loss), 5)
        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_fname = get_model_ckpt_fname(cfg, model.model_name)
            ckpt_utils.save_ckpt(cfg.run.ckpt_dir, ckpt_fname, cur_epoch + start_epoch, model, optimizer,
                                 cfg.run.MAX_CKPT_KEEP_NUM, train_loss=train_loss, val_loss=val_loss)
        
        print(f'epoch-{cur_epoch + start_epoch} \t \
                train_loss = {train_loss} \t \
                val_loss = {val_loss} \t \
                best_loss = {best_loss}')



def train_model_epoch(device, loader_train, model, optimizer, scheduler):
    model.train()
    count = 0
    avg_loss = 0

    task_self_attn_mask = torch.ones([cfg.dataset.num_task, cfg.dataset.num_task]).bool()
    task_self_attn_mask = task_self_attn_mask.to(device)

    loop_train = tqdm(loader_train, leave=False)
    for (train_X, train_Q, train_labels, train_masks) in loop_train:
        train_X = train_X.to(device)
        train_Q = train_Q.to(device)
        train_labels = train_labels.to(device)
        train_masks = train_masks.to(device)

        optimizer.zero_grad()
        pred = model(train_X, train_Q, task_self_attn_mask, train_masks.bool())
        labels, preds = get_valid_labels(train_labels, pred, train_masks)     
        
        task_type = 'regression'
        loss, _  = compute_loss(cfg.model.loss_fun, preds, labels, task_type)
        loss.backward()
        optimizer.step()

        # stats related
        # NOTE: the batch size here does not refer to the {X, Q} batch pair
        # but refers to the {X, Q, T_i} triplets
        loop_train.set_description(f"Progress")
        batch_size = labels.shape[0]
        count += batch_size
        avg_loss += loss.detach().cpu() * batch_size

    scheduler.step()
    avg_loss /= count
    torch.cuda.empty_cache()
    return avg_loss


def eval_model(device, loader_eval, model, data_for_recalls=None):
    model.eval()

    avg_loss = 0
    count = 0

    _preds = []
    with torch.no_grad():
        task_self_attn_mask = torch.ones([cfg.dataset.num_task, cfg.dataset.num_task]).bool()
        task_self_attn_mask = task_self_attn_mask.to(device)

        for _, (eval_X, eval_Q, eval_labels, eval_masks) in enumerate(loader_eval):
            eval_X = eval_X.to(device)
            eval_Q = eval_Q.to(device)
            eval_labels = eval_labels.to(device)
            eval_masks = eval_masks.to(device)

            pred = model(eval_X, eval_Q, task_self_attn_mask, eval_masks.bool())
            labels, pred = get_valid_labels(eval_labels, pred, eval_masks)

            task_type = 'regression'
            loss, _ = compute_loss(cfg.model.loss_fun, pred, labels, task_type)
            
            batch_size = labels.shape[0]
            count += batch_size
            avg_loss += loss.detach().cpu() * batch_size

            _preds.append(pred.detach().cpu())
            
    avg_loss = float(avg_loss / count)
    if data_for_recalls is not None: # visualize recall values
        print('Results on testing dataset:')
        print(f'    Loss = {round(avg_loss, 5)}')
        true_labels, masks, task_value_norm_params = data_for_recalls
        _preds = np.concatenate(_preds)
        preds = np.zeros_like(true_labels, dtype=true_labels.dtype)
        idxes = np.where(masks > 0.5)
        preds[idxes] = _preds
        preds, _ = recover_task_values(preds, masks, buffer=task_value_norm_params)
        preds = preds[idxes]
        true_labels = true_labels[idxes]
        eval_utils.visualize_err(preds, true_labels)

    torch.cuda.empty_cache()
    return avg_loss, _preds


if __name__ == '__main__':
    cfg = getConfigs()
    device = torch.device(f'cuda:{cfg.run.gpu}')
    torch.cuda.empty_cache()
    print(f'device = {device}')

    (train_data, validation_data, test_data, original_test_task_values, task_value_norm_params) = load_data(cfg)
    test_masks = test_data[-1]

    train_loader, validation_loader, _test_loader = create_loaders(train_data, validation_data, test_data, cfg.run.batch_size)
    loaders = (train_loader, validation_loader, _test_loader)
    data_for_recalls = (original_test_task_values, test_masks, task_value_norm_params)

    # TODO: This block is verbose and ugly. Try to make it more elegant.
    model_loaded = False
    print("Creating GRELA...")
    model = GRELA(cfg)
    model_start_epoch = 0
    best_loss = 1e50
    if cfg.run.eval_model:
        # print("Loading GRELA...")
        model_ckpt_fname = get_model_ckpt_fname(cfg, model.model_name)

        model, model_start_epoch, model_best_loss = ckpt_utils.load_model(cfg.run.ckpt_dir,
                                                                                 model_ckpt_fname,
                                                                                 model,
                                                                                 device=device)
        if model_start_epoch >= 0:
            print("Finished loading GRELA.")
            eval_model(device, loader_eval=loaders[2], model=model, data_for_recalls=data_for_recalls)
            model_loaded = True
        else:
            model_start_epoch = 0
            print("There is no available GRELA.")

    # this step could be skipped if models are saved
    if cfg.run.train_model:
        model.to(device)
        optimizer = create_optimizer(cfg, model.parameters())
        scheduler = create_scheduler(cfg, optimizer)
        max_epoch = cfg.optim.max_epoch
        train_model(cfg, device, loaders, model, optimizer, scheduler, max_epoch,
                    start_epoch=model_start_epoch, best_loss=best_loss)


# GRELA
# python run.py --train_model True --eval_model False --max_epoch 400 --base_lr 1e-5 --num_s_self_attn_layers 6 --num_q_cross_attn_layers 6 --num_g_attn_layers 6 --attn_head_key_dim 1024 --query_emb_dim 1024 --fix_keys_in_attn False --gpu 7
# python run.py --train_model False --eval_model True --max_epoch 400 --base_lr 1e-5 --num_s_self_attn_layers 6 --num_q_cross_attn_layers 6 --num_g_attn_layers 6 --attn_head_key_dim 1024 --query_emb_dim 1024 --fix_keys_in_attn False --gpu 7

# python run.py --train_model True --eval_model False --max_epoch 400 --base_lr 1e-5 --num_s_self_attn_layers 6 --num_q_cross_attn_layers 6 --num_g_attn_layers 6 --attn_head_key_dim 1024 --query_emb_dim 1024 --fix_keys_in_attn False --gpu 6 --wl_type dynamic
# python run.py --train_model False --eval_model True --max_epoch 400 --base_lr 1e-5 --num_s_self_attn_layers 6 --num_q_cross_attn_layers 6 --num_g_attn_layers 6 --attn_head_key_dim 1024 --query_emb_dim 1024 --fix_keys_in_attn False --gpu 6 --wl_type dynamic