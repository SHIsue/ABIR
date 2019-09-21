import numpy as np
import os
import torch
path = '../data/consumer/Eval/list_eval_partition.txt'
base = '../data/consumer/'
cat_f = open(path,'r')
cat_raw_data = cat_f.readlines()[2:]

print(len(cat_raw_data))


for line in cat_raw_data:
    if(line[-1]=='\n'):
        line = line[:-1]


count = 0

train= []
val_q = []
test_q = []
val_g = []
test_g = []



for i in range(len(cat_raw_data)):
    cat_line = cat_raw_data[i].split()


    q = cat_line[0]
    g = cat_line[1]
    id = str(int(cat_line[2][3:]))
    status = cat_line[3]
    res_q = q + ',' + id
    res_g = g + ',' + id
    if(status == 'train'):
        train.append(res_q)
        train.append(res_g)
    elif(status=='val'):
        val_q.append(res_q)
        val_g.append(res_g)
    else:
        test_q.append(res_q)
        test_g.append(res_g)


a = open(base+'train.txt','w')
a.write('\n'.join(train))
b = open(base+'val_q.txt','w')
b.write('\n'.join(val_q))
c = open(base+'test_q.txt','w')
c.write('\n'.join(test_q))
e = open(base+'val_g.txt','w')
e.write('\n'.join(val_g))
f = open(base+'test_g.txt','w')
f.write('\n'.join(test_g))


# # f = open(path,'r')
# # lines = f.readlines()[2:]
# # img_list = []
# # labels = []
# # train = []
# # query = []
# # gallery = []
# # for line in lines:
# #     data = line[:-1].split()
# #     if(data[2]=='train'):
# #         train.append(data[0]+','+str(int(data[1][3:])))
# #     elif(data[2]=='query'):
# #         query.append(data[0]+','+str(int(data[1][3:])))
# #     else:
# #         gallery.append(data[0]+','+str(int(data[1][3:])))
# #
# # a = open(base+'train.txt','w')
# # a.write('\n'.join(train))
# # b = open(base+'gallery.txt','w')
# # b.write('\n'.join(gallery))
# # c = open(base+'query.txt','w')
# # c.write('\n'.join(query))
#
# f.close()
