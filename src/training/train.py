import json
import logging
import math
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss, Clip_DA_all_hn_far_Loss, Siglip_DA_all_hn_far_Loss
from .distributed import is_master
from .zero_shot import zero_shot_eval


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch_NegCLIP(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, new_images, texts, new_texts, hard_captions, new_hard = batch

        images = images.to(device=device, non_blocking=True)
        new_images = new_images.to(device=device, non_blocking=True)

        texts = texts.to(device=device, non_blocking=True)
        new_texts = new_texts.to(device=device, non_blocking=True)

        hard_captions = hard_captions.to(device=device, non_blocking=True)
        new_hard = new_hard.to(device=device, non_blocking=True)

        images = torch.cat([images, new_images])

        texts = torch.cat([texts, new_texts])
        texts = torch.cat([texts, hard_captions])
        texts = torch.cat([texts, new_hard])

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} - V4"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def train_one_epoch_all_hn_far(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()
    
    loss = Clip_DA_all_hn_far_Loss(
        local_loss = args.local_loss, 
        gather_with_grad = args.gather_with_grad,
        cache_labels = True,
        rank = args.rank,
        world_size = args.world_size,
        use_horovod = args.horovod,
        cmr_loss = args.use_cmr_loss, 
        imc_loss = args.use_imc_loss, 
        hndc_loss = args.use_hndc_loss,
        hardnegative = True, 
        imc_loss_weight = args.imc_loss_weight, 
        cmr_loss_weight = args.cmr_loss_weight, 
        hndc_loss_weight = args.hndc_loss_weight,
        threshold_type = "mean"
    )

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    #threshold 초기화
    thresholds = 0.0
    th_upper_bound = 10.0
    


    #loss들 초기화
    cmr_loss = None
    imc_loss = None
    hndc_loss = None
    itc_loss = None
    
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, texts, valid_hn_masks = batch

        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        valid_hn_masks = valid_hn_masks.to(device=device, non_blocking=True)
        # print(f"images shape : {images.shape}")
        # print(f"texts shape : {texts.shape}")


        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            # total_loss = loss(image_features, text_features, logit_scale)
            total_loss, thresholds, itc_loss, cmr_loss, imc_loss, hndc_loss = loss(image_features, text_features, valid_hn_masks, logit_scale, thresholds)
            #Threshold upper bound
            # thresholds=torch.clamp(thresholds,0,th_upper_bound)
            if args.use_cmr_loss and thresholds is not None:
                thresholds = torch.clamp(thresholds, 0, th_upper_bound)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} - V4 "
                f"ITC_Loss: {itc_loss.item() if itc_loss is not None else 'N/A'} "
                f"CMR_Loss: {cmr_loss.item() if cmr_loss is not None else 'N/A'} "
                f"IMC_Loss: {imc_loss.item() if imc_loss is not None else 'N/A'} "
                f"HNDC_Loss: {hndc_loss.item() if hndc_loss is not None else 'N/A'}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "CMR_Loss": cmr_loss if cmr_loss is not None else None,
                "IMC_Loss": imc_loss if imc_loss is not None else None,
                "HNDC_Loss": hndc_loss if hndc_loss is not None else None,
                "ITC_Loss": itc_loss if itc_loss is not None else None   # 추가
            }
            if args.use_cmr_loss and thresholds is not None:
                log_data["threshold_1"] = thresholds.mean(0)[0].item()
                log_data["threshold_2"] = thresholds.mean(0)[1].item()
                log_data["threshold_3"] = thresholds.mean(0)[2].item()
                log_data["threshold_4"] = thresholds.mean(0)[3].item()
                        
            
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def train_one_epoch_None_neg(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, texts, _ = batch
        
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        # hard_captions = hard_captions.to(device=device, non_blocking=True)
        
        # texts = torch.cat([texts, hard_captions])

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} - V4"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def train_one_epoch_all_hn_far_siglip(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress

    model.train()
    loss = Siglip_DA_all_hn_far_Loss(
        local_loss = args.local_loss, 
        gather_with_grad = args.gather_with_grad,
        cache_labels = True,
        rank = args.rank,
        world_size = args.world_size,
        use_horovod = args.horovod,
        cmr_loss = args.use_cmr_loss, 
        imc_loss = args.use_imc_loss, 
        hndc_loss = args.use_hndc_loss,
        hardnegative = args.use_hardnegative, 
        imc_loss_weight = args.imc_loss_weight, 
        cmr_loss_weight = args.cmr_loss_weight, 
        hndc_loss_weight = args.hndc_loss_weight,
        threshold_type = "mean"
    ) 

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    
    #threshold 초기화
    thresholds = 0.0
    th_upper_bound = 10.0
    


    #loss들 초기화
    cmr_loss = None
    imc_loss = None
    hndc_loss = None
    itc_loss = None
    
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, texts, valid_hn_masks = batch

        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        valid_hn_masks = valid_hn_masks.to(device=device, non_blocking=True)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            # image_features, text_features, logit_scale = model(images, texts)
            #multi gpu
            # image_features = model.module.get_image_features(images)
            # text_features = model.module.get_text_features(texts)
            # logit_scale = model.module.logit_scale.exp()

            #single gpu
            text_features = model.get_text_features(texts)
            image_features = model.get_image_features(images)
            logit_scale = model.logit_scale.exp()
            
            total_loss, thresholds, itc_loss, cmr_loss, imc_loss, hndc_loss = loss(image_features, text_features, valid_hn_masks, logit_scale, thresholds)
            #Threshold upper bound
            # thresholds=torch.clamp(thresholds,0,th_upper_bound)
            if args.use_cmr_loss and thresholds is not None:
                thresholds = torch.clamp(thresholds, 0, th_upper_bound)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} - V4 "
                f"ITC_Loss: {itc_loss.item() if itc_loss is not None else 'N/A'} "
                f"CMR_Loss: {cmr_loss.item() if cmr_loss is not None else 'N/A'} "
                f"IMC_Loss: {imc_loss.item() if imc_loss is not None else 'N/A'} "
                f"HNDC_Loss: {hndc_loss.item() if hndc_loss is not None else 'N/A'}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "CMR_Loss": cmr_loss if cmr_loss is not None else None,
                "IMC_Loss": imc_loss if imc_loss is not None else None,
                "HNDC_Loss": hndc_loss if hndc_loss is not None else None,
                "ITC_Loss": itc_loss if itc_loss is not None else None   # 추가
            }
            if args.use_cmr_loss and thresholds is not None:
                log_data["threshold_1"] = thresholds.mean(0)[0].item()
                log_data["threshold_2"] = thresholds.mean(0)[1].item()
                log_data["threshold_3"] = thresholds.mean(0)[2].item()
                log_data["threshold_4"] = thresholds.mean(0)[3].item()
                        
            
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, hard_images, texts, texts_hard_images, hard_captions, hard_captions_of_hard_images = batch

                images = images.to(device=device, non_blocking=True)
                hard_images = hard_images.to(device=device, non_blocking=True)

                texts = texts.to(device=device, non_blocking=True)
                texts_hard_images = texts_hard_images.to(device=device, non_blocking=True)

                hard_captions = hard_captions.to(device=device, non_blocking=True)
                hard_captions_of_hard_images = hard_captions_of_hard_images.to(device=device, non_blocking=True)

                images = torch.cat([images, hard_images])

                texts = torch.cat([texts, texts_hard_images])
                texts = torch.cat([texts, hard_captions])
                texts = torch.cat([texts, hard_captions_of_hard_images])

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic

                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()
                    
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features[:len(logits_per_image)].cpu())

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text[:len(logits_per_image)], labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text[:len(logits_per_image)]}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


