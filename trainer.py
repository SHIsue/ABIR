# coding=utf-8
from __future__ import print_function, absolute_import
import time
from utils import AverageMeter, orth_reg
import  torch
from torch.autograd import Variable
from torch.backends import cudnn

cudnn.benchmark = True


def train(writer,epoch, model, criterion, optimizer, train_loader, args,):
    #total_losses = AverageMeter()
    losses = AverageMeter()
    #addition_loss = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()

    end = time.time()
    steps = epoch * len(train_loader) * args.batch_size
    freq = min(args.print_freq, len(train_loader))

    for i, data_ in enumerate(train_loader, 0):
        inputs, labels= data_
        # wrap them in Variable
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()


        optimizer.zero_grad()

        outputs= model(inputs)
        # attention_center_loss = criterion2(center_feat)
        loss, inter_, dist_ap, dist_an = criterion(outputs[0], labels)
        # if(args.use_reg):
        #     orth_reg_loss = orth_reg(model,0.1)
        #     loss+=orth_reg_loss
        # total_loss = loss + (loss1+loss2+loss3)/3
        #total_loss = loss + div_loss
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #total_losses.update(total_loss.item())
        #addition_loss.update(div_loss.item())
        losses.update(loss.item())
        accuracy.update(inter_)
        pos_sims.update(dist_ap)
        neg_sims.update(dist_an)
        # if  args.use_reg and ( (i + 1) % freq == 0 or (i+1) == len(train_loader)):
        #     print('orth_reg_loss: {:.4f}\t'.format(orth_reg_loss.item()))
        if (i + 1) % freq == 0 or (i+1) == len(train_loader):
            print('Epoch: [{0:03d}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Accuracy {accuracy.avg:.4f}\t'
                  'Pos {pos.avg:.4f}\t'
                  'Neg {neg.avg:.4f}\t'.format
                  (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                   loss=losses, accuracy=accuracy, pos=pos_sims, neg=neg_sims))
            writer.add_scalar('loss', losses.avg, global_step=steps)
            # writer.add_scalar('div_loss', addition_loss.avg, global_step=steps)
            # writer.add_scalar('total_loss', total_losses.avg, global_step=steps)
            writer.add_scalar('accuracy', accuracy.avg, global_step=steps)
            writer.add_scalar('pos_sims', pos_sims.avg, global_step=steps)
            writer.add_scalar('neg_sims', neg_sims.avg, global_step=steps)
            # addition_loss.reset()
            # total_losses.reset()
            losses.reset()
            accuracy.reset()
            pos_sims.reset()
            neg_sims.reset()

        if epoch == 0 and i == 0:
            print('-- HA-HA-HA-HA-AH-AH-AH-AH --')
        steps+=args.batch_size