# Train the model
import sys
import os
from os.path import dirname as up
import torch
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist

from utils.metrics import Metrics
from utils.loss import Loss
from utils.schedulers import get_scheduler
from utils.optimizers import get_optimizer
from utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from datasets import PatchSet
from models import FSDFormer

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


def main(cfg, gpu, save_dir):

    start = time.time()

    best_ssim = 0.0

    # device
    device = torch.device(cfg['DEVICE'])
    
    train_cfg = cfg['TRAIN']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    optim_cfg, sched_cfg = cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    num_workers = train_cfg['NUM_WORKERS']
    
    # dataset
    trainset = PatchSet(dataset_cfg['DATES_TRAIN'], dataset_cfg['TRAIN'], 'train', train_cfg['IMAGE_SIZE'], train_cfg['PATCH_SIZE'])
    valset = PatchSet(dataset_cfg['DATES_VAL'], dataset_cfg['VAL'], 'val', train_cfg['IMAGE_SIZE'], train_cfg['PATCH_SIZE'])

    model = FSDFormer()

    model = model.to(device)

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[gpu])
    elif train_cfg['DP']:
        sampler = RandomSampler(trainset)
        model = torch.nn.DataParallel(model)
    else:
        sampler = RandomSampler(trainset)
        model = model
    
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=False, sampler=sampler)
    valloader = DataLoader(valset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=False)

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    loss_fn = Loss()
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs * iters_per_epoch, sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(str(save_dir / 'logs'))

    for epoch in range(epochs):
        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0

        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, (C2, F2, C1, F1, gt_mask) in pbar:
            optimizer.zero_grad(set_to_none=True)

            F1 = F1.to(device)
            F2 = F2.to(device)
            C1 = C1.to(device)
            C2 = C2.to(device)
            gt_mask = gt_mask.to(device)
            
            with autocast(enabled=train_cfg['AMP']):

                result = model(F1, C2, C1)

                loss = loss_fn(result, F2, is_ds=False)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        
        train_loss /= iter+1
        writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()

        if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            cur_result = evaluate(model, valloader, device, cfg)
            rmse = np.mean(np.array(cur_result['rmse']))
            ssim = np.mean(np.array(cur_result['ssim']))
            ergas = np.mean(np.array(cur_result['ergas']))
            psnr = np.mean(np.array(cur_result['psnr']))

            writer.add_scalar('val/rmse', rmse, epoch)
            writer.add_scalar('val/ssim', ssim, epoch)
            writer.add_scalar('val/ergas', ergas, epoch)
            writer.add_scalar('val/psnr', psnr, epoch)

            print('RMSE: %.4f SSIM: %.4f ERGAS: %.4f PSNR: %.4f' % (
                rmse, ssim, ergas, psnr))
   
            if ssim > best_ssim:
                best_ssim = ssim
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")
            print(f"Current ssim: {ssim} Best IoU: {best_ssim}")
            if (epoch+1) == epochs:
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}+_final.pth")

    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best acc', f"{best_ssim:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    
    print(tabulate(table, numalign='right'))

@torch.no_grad()
def evaluate(model, dataloader, device, cfg):
    print('Evaluating...')
    model.eval()
    metrics = Metrics() 
    
    for C2, F2, C1, F1, gt_mask in tqdm(dataloader):
        F1 = F1.to(device) 
        F2 = F2.to(device)
        C1 = C1.to(device)
        C2 = C2.to(device)
        gt_mask = gt_mask.to(device)
        
        result = model(F1, C2, C1)

        metrics.update(result, F2)
        c_results = metrics.compute()

    return c_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/config.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407) # Set seed for reproducibility
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    os.makedirs(save_dir, exist_ok=True)
    main(cfg, gpu, save_dir)
    cleanup_ddp()