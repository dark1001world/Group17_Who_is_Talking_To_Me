import os, logging
import time
import torch
import torch.optim
import torch.utils.data
from common.utils import AverageMeter
from common.utils import save_checkpoint, PostProcessor
from common.distributed import is_master
import common.lr_sched as lr_sched

from common.utils import PostProcessor

logger = logging.getLogger(__name__)


def train(train_loader, model, criterion, optimizer, epoch, args=None):
    logger.info('training')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()

    model.train()

    end = time.time()

    for i,  (source_frame, target) in enumerate(train_loader):

        if args.model == 'VideoMAEv2':
            # we use a per iteration (instead of per epoch) lr scheduler for VideoMAEv2
            lr = lr_sched.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)
        else:
            lr = optimizer.param_groups[0]['lr']

        # measure data loading time
        data_time.update(time.time() - end)
        source_frame = source_frame.cuda()
        target = target.cuda()

        # compute output
        output = model(source_frame)

        if args.focal_loss:
            x = output[:, 1:]
            loss = criterion(x, target)
        else:
            target = target.squeeze(1)
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 10 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Lr {lr:.6f}\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=avg_loss,
                        lr=lr))


def validate(val_loader, model, postprocess, mode='val', args=None):
    logger.info('evaluating')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()
    end = time.time()

    for i, (source_frame, target) in enumerate(val_loader):

        # measure data loading time
        data_time.update(time.time() - end)       
        source_frame = source_frame.cuda()

        with torch.no_grad():
            output = model(source_frame)
            postprocess.update(output.detach().cpu(), target)

            batch_time.update(time.time() - end)
            end = time.time()

        if i % 10 == 0:
            logger.info('Processed: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        i, len(val_loader), batch_time=batch_time, data_time=data_time))
    postprocess.save()

    if mode == 'val':
        mAP = None
        if is_master():
            mAP = postprocess.get_mAP()
        return mAP

    if mode == 'test':
        print('generate pred.csv')

def infer(val_loader, model, postprocess, mode='infer', args=None):
    logger.info('evaluating')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()
    end = time.time()
    gg = 0
    for i, (source_frame, target) in enumerate(val_loader):

        data_time.update(time.time() - end)
        source_frame = source_frame.cuda()
   
        with torch.no_grad():
            output = model(source_frame)
            # print(output)
            postprocess.update(output.detach().cpu(), target)

            batch_time.update(time.time() - end)
            end = time.time()
        if i % 10 == 0:
            logger.info('Processed: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        i, len(val_loader), batch_time=batch_time, data_time=data_time))
    postprocess.save()

    if mode == 'infer':
        print('infer_pred.csv')
        
        
def train_landmark(train_loader, model, criterion, optimizer, epoch, args=None):
    logger.info('training')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()

    model.train()

    end = time.time()

    for i,  (source_frame, target, face_landmark) in enumerate(train_loader):

        if args.model == 'VideoMAEv2':
            # we use a per iteration (instead of per epoch) lr scheduler for VideoMAEv2
            lr = lr_sched.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)
        else:
            lr = optimizer.param_groups[0]['lr']

        # measure data loading time
        data_time.update(time.time() - end)
        source_frame = source_frame.cuda()
        target = target.cuda()
        face_landmark = face_landmark.cuda()

        # compute output
        output = model(source_frame, face_landmark = face_landmark)

        if args.focal_loss:
            x = output[:, 1:]
            loss = criterion(x, target)
        else:
            target = target.squeeze(1)
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.update(loss.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 100 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Lr {lr:.6f}\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=avg_loss,
                        lr=lr))
            
        if i%4000==0 and i!=0:
            postprocess_val = PostProcessor(args)
            val_params['postprocess']=postprocess_val
            with torch.no_grad():
                mAP,loss = validate(**val_params)
            model.train()

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            logger.info(f'mAP: {mAP:.4f} best mAP: {best_mAP:.4f}')
            if is_best:
                save_checkpoint({
                    'epoch': global_step,
                    'state_dict': model.state_dict(),
                    'mAP': mAP},
                    save_path=args.exp_path,
                    is_best=is_best,
                    is_dist=args.dist)
    postprocess.save()

                

def validate_landmark(val_loader, model, postprocess, mode='val'):
    logger.info('evaluating')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()
    end = time.time()

    for i, (source_frame, target, face_landmark) in enumerate(val_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        source_frame = source_frame.cuda()
        face_landmark = face_landmark.cuda()

        with torch.no_grad():
            output = model(source_frame, face_landmark)
            postprocess.update(output.detach().cpu(), target)

            batch_time.update(time.time() - end)
            end = time.time()

        if i % 10 == 0:
            logger.info('Processed: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                        i, len(val_loader), batch_time=batch_time, data_time=data_time))
    postprocess.save()

    if mode == 'val':
        mAP = None
        if is_master():
            mAP = postprocess.get_mAP()
        return mAP

    if mode == 'test':
        print('generate pred.csv')