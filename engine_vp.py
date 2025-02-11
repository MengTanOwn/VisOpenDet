import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 
from misc import (MetricLogger, SmoothedValue, reduce_dict)
def train_one_epoch_vp(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, weight_dict:dict = {},**kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)
    early_stop = 0
    for sample_query, visual_prompts, targets in metric_logger.log_every(data_loader, print_freq, header):
        # early_stop +=1
        # if early_stop>5000:
        #     break
        sample_query = sample_query.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for t in visual_prompts:
            t['boxes'] = t['boxes'].to(device)
            t['cates'] = t['cates'].to(device)
        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(sample_query,targets=targets,vp=visual_prompts)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs,targets=targets,vp=visual_prompts)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(sample_query,targets=targets,vp=visual_prompts)
            # temp = model.module.decoder.enc_score_head.lang_log_scale[0].exp()
            loss_dict = criterion(outputs, targets)
                   
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_value = sum(loss_dict_reduced_scaled.values())
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        loss_dict_reduced = {k:v for k,v in loss_dict_reduced.items() if 'aux' not in k and 'dn' not in k}
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # print("Gradient norms:", [p.grad.norm().item() if p.grad is not None else None for p in model.parameters()])
        # print("Weight norms:", [p.norm().item() for p in model.parameters()])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}