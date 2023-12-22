from . import FileViewer
import os
import torch
import sys
sys.path.append("../../")


def load_model(ckpt_dir, ckpt_fname, model, device, allow_partial_load=False):
    ckpt_paths = FileViewer.list_files(ckpt_dir, suffix='pt', prefix=f'{ckpt_fname}-', isdepth=False)
    latest_epoch = -1
    latest_path = None
    for path in ckpt_paths:
        fname = os.path.basename(path)[0:-3]
        terms = fname.split('-')
        epoch_no = int(terms[-1])
        if epoch_no > latest_epoch:
            latest_epoch = epoch_no
            latest_path = path
    if latest_path is None:
        return model, -1, 1e50
    # assert latest_path is not None

    print(f"Loading the saved model...")
    try:
        checkpoint = torch.load(latest_path, map_location=device)
    except Exception as e:
        print(type(e))
        print(e)
        print("Some issue occured when loading the saved model.")
        print("Please check the file path.")
        return None


    trained_dict = checkpoint['model_state_dict']

    if allow_partial_load:
        model_dict = model.state_dict()
        trained_dict = {k: v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(trained_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(trained_dict)
    start_epoch = checkpoint['epoch'] + 1
    if 'val_loss' in checkpoint:
        best_loss = checkpoint['val_loss']
    else:
        best_loss = checkpoint['train_loss']
    model.to(torch.device(device))
    print(f"Finished loading the saved model.")

    # print('-' * 50)
    # for k, v in trained_dict.items():
    #     print(k)
    # print('-' * 50)
    return model, start_epoch, best_loss


def remove_extra_ckpts(ckpt_dir, ckpt_fname, MAX_CKPT_KEEP_NUM):
    ckpt_paths = FileViewer.list_files(ckpt_dir, suffix='pt', prefix=f'{ckpt_fname}-', isdepth=False)
    if len(ckpt_paths) > MAX_CKPT_KEEP_NUM:
        epoch_nos = []
        for path in ckpt_paths:
            fname = os.path.basename(path)[0:-3]
            terms = fname.split('-')
            epoch_no = int(terms[-1])
            epoch_nos.append(epoch_no)
        epoch_nos.sort()
        n = len(epoch_nos) - MAX_CKPT_KEEP_NUM
        for i in range(n):
            epoch_no = epoch_nos[i]
            file_path = os.path.join(ckpt_dir, f"{ckpt_fname}-{epoch_no}.pt")
            os.remove(file_path)


def save_ckpt(ckpt_dir, ckpt_fname, cur_epoch, model, optimizer, MAX_CKPT_KEEP_NUM, **kwargs):
    file_path = os.path.join(ckpt_dir, f"{ckpt_fname}-{cur_epoch}.pt")
    # print('file_path =', file_path)
    save_dict = {
        'epoch': cur_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    for key in kwargs:
        save_dict[key] = kwargs[key]

    torch.save(save_dict, file_path)
    remove_extra_ckpts(ckpt_dir, ckpt_fname, MAX_CKPT_KEEP_NUM)
