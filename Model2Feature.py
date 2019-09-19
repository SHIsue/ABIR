# coding=utf-8


import torch
import torch.utils
import torch.utils.data
from torch.backends import cudnn
from evaluations import extract_features
import models
import DataSet
from utils.serialization import load_checkpoint
cudnn.benchmark = True


def Model2Feature(data, model,  nThreads=16, batch_size=32, pool_feature=False, **kargs):
    dataset_name = data

    data = DataSet.create(data)

    if dataset_name in ['inshop', 'jd_test']:
        gallery_loader = torch.utils.data.DataLoader(
            data.gallery, batch_size=batch_size, shuffle=False,
            drop_last=False)


        query_loader = torch.utils.data.DataLoader(
            data.query, batch_size=batch_size,
            shuffle=False, drop_last=False)

        gallery_feature, gallery_labels = extract_features(model, gallery_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        query_feature, query_labels = extract_features(model, query_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)

    else:
        data_loader = torch.utils.data.DataLoader(
            data.test, batch_size=batch_size,
            shuffle=False, drop_last=False,
            num_workers=nThreads)
        features, labels = extract_features(model, data_loader, print_freq=1e5, metric=None, pool_feature=pool_feature)
        gallery_feature, gallery_labels = query_feature, query_labels = features, labels
    return gallery_feature, gallery_labels, query_feature, query_labels

