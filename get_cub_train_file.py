'''
    author: Xinyao Nie, Zijian Wang
    date: 26/09/2019
    github: https://github.com/SHIsue/ABIR
'''

img_path = '../data/cub/images.txt' #image path
label_path = '../data/cub/image_class_labels.txt'
train_split_path = '../data/cub/train_test_split.txt'

f_imgs = open(img_path,'r')
f_labels = open(label_path,'r')
f_is_train = open(train_split_path,'r')
# train is 1; test is 0
imgs_data = f_imgs.read().split('\n')[:-1]
# meet \n then split
labels_data = f_labels.read().split('\n')[:-1]
is_train_data = f_is_train.read().split('\n')[:-1]
imgs = []
train = []
test = []

for i,line in enumerate(imgs_data):
    _,img= line.split(' ')
    _,label = labels_data[i].split(' ')
    _,is_train = is_train_data[i].split(' ')
    res = img+' '+label
    if(is_train=='1'):
        train.append(res)
    else:
        test.append(res)

a = open('../data/cub/train.txt','w')
a.write('\n'.join(train))
b = open('../data/cub/test.txt','w')
b.write('\n'.join(test))


