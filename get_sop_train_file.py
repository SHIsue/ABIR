train_path = '../data/sop/Ebay_train.txt'
test_path = '../data/sop/Ebay_test.txt'
base = '../data/sop/'
f_train = open(train_path,'r')
f_test = open(test_path,'r')
train_data = f_train.read().split('\n')[1:-1]
test_data = f_test.read().split('\n')[1:-1]
print(train_data[-1])
print(test_data[0])
train = []
test = []
for line in train_data:
    _,id,_,img= line.split(' ')
    res = img+' '+id
    train.append(res)
for line in test_data:
    _,id,_,img= line.split(' ')
    res = img+' '+id
    test.append(res)
a = open(base+'train.txt','w')
a.write('\n'.join(train))
b = open(base+'test.txt','w')
b.write('\n'.join(test))

# print(len(train))
# print(len(test))

