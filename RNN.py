import numpy as np
import keras
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
np.random.seed(420)

temp1=open("Dataset/adjectives.txt","r")
temp2=open("Dataset/verbs.txt","r")
adj=open("Dataset/finaladj1.txt","a")
ver=open("Dataset/finalver1.txt","a")
for i in temp1: adj.write(i.lower())
for i in temp2:ver.write(i.lower())

with open('Dataset/finaladj1.txt', 'r') as f:
    text=f.read()
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))

from random import randint
X_train = []
Y_train=  []
X_test = []
Y_test=[]
a=open("Dataset/finaladj1.txt","r")
for word in a:
    temp=[]
    for letter in word:
        temp.append(vocab_to_int[letter])
    for i in range(25-len(temp)):
        temp.append(0)
    temp=np.asarray(temp)
    x=float(randint(1,100)/100)
    if x<=0.2:
        X_test.append(temp)
        Y_test.append(0)
    else:
        Y_train.append(0)
        X_train.append(temp)
a=open("Dataset/finalver1.txt","r")
for word in a:
    temp=[]
    for letter in word:
        temp.append(vocab_to_int[letter])
    for i in range(25-len(temp)):
        temp.append(0)
    temp=np.asarray(temp)
    x=float(randint(1,100)/100)
    if x<=0.2:
        X_test.append(temp)
        Y_test.append(1)
    else:
        Y_train.append(1)
        X_train.append(temp)
    #Y.append(1)
    #X.append(temp)
X_test=np.asarray(X_test)
Y_test= np.asarray(Y_test)
X_train=np.asarray(X_train)
Y_train= np.asarray(Y_train)

model = Sequential()
model.add(Embedding(input_dim = 22883, output_dim = 50, input_length = 25))
model.add(LSTM(output_dim=16, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(output_dim=16, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, validation_split = 0.1, verbose = 1)
score, acc = model.evaluate(X_test, Y_test, batch_size=1)
print('Test score:', score)
print('Test accuracy:', acc)
