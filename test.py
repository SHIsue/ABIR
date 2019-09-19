# coding=utf-8
from __future__ import absolute_import, print_function
import argparse
from Model2Feature import Model2Feature
from evaluations import Recall_at_ks, pairwise_similarity
from utils.serialization import load_checkpoint
import torch 
from tensorboardX import SummaryWriter
from config import Config


def eval(ckp_path=None,model = None):
    args = Config()
    if(ckp_path!=None):
        checkpoint = load_checkpoint(ckp_path,args)
    else:
        checkpoint = model
        checkpoint.eval()
    # print(args.pool_feature)

    gallery_feature, gallery_labels, query_feature, query_labels = \
        Model2Feature(data=args.data,model = checkpoint, batch_size=args.batch_size, nThreads=args.nThreads, pool_feature=args.pool_feature)

    sim_mat = pairwise_similarity(query_feature, gallery_feature)
    if args.gallery_eq_query is True:
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))

    recall_ks = Recall_at_ks(sim_mat, query_ids=query_labels, gallery_ids=gallery_labels, data=args.data)
    if(ckp_path==None):
        checkpoint.train()
    return recall_ks

if __name__ =='__main__':
    args = Config()
    writer = SummaryWriter('log/'+args.log_name)
    for i in range(0, 1):
        epoch = 30* (33-i)
        #ckp_path = 'saved_model/ckp_ep%s.pth.tar' % str(epoch)
        ckp_path ='saved_model/ckp_ep810.pth.tar'
        res = eval(ckp_path)
        writer.add_scalar('top1', res[0], global_step=epoch)
        writer.add_scalar('top10', res[1], global_step=epoch)
        writer.add_scalar('top20', res[2], global_step=epoch)
        writer.add_scalar('top50', res[-1], global_step=epoch)

