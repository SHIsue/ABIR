
from __future__ import absolute_import, print_function

import torch.utils.data
import DataSet

from torch.backends import cudnn
import models
import losses
from utils import FastRandomIdentitySampler, mkdir_if_missing, logging, display
from utils.serialization import save_checkpoint, load_checkpoint
from trainer import train
from tensorboardX import SummaryWriter

import numpy as np
import os.path as osp
from config import Config

from test import eval
cudnn.benchmark = True

use_gpu = True

# Batch Norm Freezer : bring 2% improvement on CUB 
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def main(args):

    # s_ = time.time()
    save_dir = args.save_dir
    mkdir_if_missing(save_dir)
    #sys.stdout = logging.Logger(os.path.join(save_dir, 'log.txt'))
    writer = SummaryWriter('log/'+args.log_name)
    display(args)
    start = 0

    model = models.create(args.net, pretrain=True, dim=args.dim)

    # for vgg and densenet
    if args.resume is None:
        model_dict = model.state_dict()

    else:
        # resume model
        print('load model from {}'.format(args.resume))
        model = load_checkpoint(args.resume,args)
        start = 80

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    #freeze vgg layers

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    criterion = losses.create(args.loss, margin=args.margin, alpha=args.alpha,beta = args.beta).cuda()

    data = DataSet.create(args.data)

    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.batch_size,
        sampler=FastRandomIdentitySampler(data.train, num_instances=args.num_instances),
        drop_last=True, pin_memory=True)

    # save the train information

    for epoch in range(start, args.epochs):

        # if epoch == 5:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/100)
        #     print(args.lr/100)


        train(writer,epoch=epoch, model=model, criterion=criterion,
              optimizer=optimizer, train_loader=train_loader, args=args,)

        if epoch == 800:
            optimizer = torch.optim.Adam(model.parameters(),  lr=args.lr/10, weight_decay=args.weight_decay)





        if (epoch+1) % args.save_step == 0:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'epoch': (epoch+1),
            }, is_best=False, fpath=osp.join(args.save_dir, 'ckp_ep' + str(epoch + 1) + '.pth.tar'))

        #ckp_path = 'saved_model/ckp_ep%s.pth.tar' % str(epoch + 1)
        #try:
        #     recall_ks = eval(ckp_path)
        #     writer.add_scalar('top1', recall_ks[0], global_step=epoch)
        #     writer.add_scalar('top10', recall_ks[1], global_step=epoch)
        #    writer.add_scalar('top20', recall_ks[3], global_step=epoch)
        #    writer.add_scalar('top50', recall_ks[-1], global_step=epoch)
        #except:
        #     pass

if __name__ == '__main__':
    args = Config()
    main(args)




