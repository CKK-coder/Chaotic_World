import csv
import os
fs = os.listdir('/opt/data/private/Chaotic_World/dataset/frames_320x180')
train = '/opt/data/private/Chaotic_World/annotations/AR_ava_format/chaos_train_1fps.csv'
test = '/opt/data/private/Chaotic_World/annotations/AR_ava_format/chaos_test_1fps.csv'
t = set()
a1 = set()
a2 = set()
with open(train, 'r') as f:
    reader = csv.reader(f)
    for l in reader:
        t.add(l[0])
        a1.add(l[6])
print(a1)
print(len(list(a1)))
with open(test, 'r') as f:
    reader = csv.reader(f)
    for l in reader:
        t.add(l[0])
        a2.add(l[6])
print(a2)

for f in t:
    if f not in fs:
        print(f)
    