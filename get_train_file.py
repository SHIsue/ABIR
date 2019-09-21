import numpy as np
import os
import torch
path = '../data/inshop/Eval/list_eval_partition.txt'
lm_path = '../data/inshop/Anno/list_landmarks_inshop.txt'
base = '../data/inshop/'
cat_f = open(path,'r')
lm_f = open(lm_path,'r')
cat_raw_data = cat_f.readlines()[2:]
lm_raw_data = lm_f.readlines()[2:]

print(len(cat_raw_data))
print(len(lm_raw_data))
cat_raw_data = sorted(cat_raw_data)
lm_raw_data = sorted(lm_raw_data)

for line in cat_raw_data:
    if(line[-1]=='\n'):
        line = line[:-1]
for line in lm_raw_data:
    if(line[-1]=='\n'):
        line = line[:-1]

count = 0

train = []
query = []
gallery = []



for i in range(len(cat_raw_data)):
    cat_line = cat_raw_data[i].split()
    lm_line = lm_raw_data[i].split()[3:]
    lm_data = lm_line
    lm = []

    name = cat_line[0]
    id = str(int(cat_line[1][3:]))
    status = cat_line[2]
    for i in range(len(lm_data)//3):
        lm+=lm_data[3*i+1:3*i+3]
    for i in range(8-len(lm_data)//3):
        lm+=['0','0']
    res_line = ','.join([name,id]+lm)
    print(res_line)
    if(status == 'train'):
        train.append(res_line)
    elif(status=='query'):
        query.append(res_line)
    else:
        gallery.append(res_line)


a = open(base+'train.txt','w')
a.write('\n'.join(train))
b = open(base+'gallery.txt','w')
b.write('\n'.join(gallery))
c = open(base+'query.txt','w')
c.write('\n'.join(query))


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
