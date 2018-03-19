
import json
import nltk
import gensim
import pickle
import numpy as np
from gensim import corpora, models, similarities
from sklearn.model_selection import train_test_split
import theano
from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.core import Dense,Dropout
from keras.layers import Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
theano.config.optimizer = "None"
import pandas as pd
#from keras.backend import manual_variable_initialization
#manual_variable_initialization(True)



#model=gensim.models.Word2Vec(tok_corp,min_count=1,size=32)
#model.wv.save_word2vec_format('model.bin', binary=True)
mod = gensim.models.Word2Vec.load('word2vec.bin');
#mod = gensim.models.Word2Vec.load('model.bin');
###############################################
#getting x and y from our iob data set

f = open("gstlstm1.txt", "r")
atis_dataset = f.read()
f.close()
x=[]
y=[]
f=atis_dataset.split("\n")
for i in f:
    if i!='':
        l=i.split("\t")
        x.append(l[0])
        y.append(l[1])
y=pd.Series(y)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#############################################################################
#get iob tag dic from pickle formatidx2la = {label2idx[k]: k for k in label2idx}

###############################################################################
tok_x = []
for i in range(len(x)):
    tok_x.append(x[i].split())
    

#print len(tok_x[14]), '------', tok_y[14]

vec_x = []
for sent in tok_x:
    sentvec = [mod[w] for w in sent if w in mod.vocab]
    vec_x.append(sentvec)

#vec_y will contain the iabel id for each label


"""sentend = np.zeros((300,), dtype=np.float32)
for tok_sent in vec_x:
   tok_sent[45:] = [] 
   tok_sent.append(sentend)
for tok_sent in vec_x:
    if len(tok_sent) < 46:
        for i in range(46 - len(tok_sent)):
            tok_sent.append(sentend)
from keras.preprocessing.sequence import pad_sequences
vec_y=pad_sequences(vec_y,padding='post',value=144)"""

 

        
#print type(vec_x[0]),' ----------- ',vec_y[0]
#preprocessing is done
###############################################################################

#below line is giving error even i have just copied the complete code from semicolon's work for x(input to model)
#vec_x = np.array(vec_x, dtype=np.float64)
#label_y = np.array(label_y, dtype=np.float64)
"""x_train=vec_x[:3984,:,:]
x_test=vec_x[3984:,:,:]
y_train=label_y[:3984,:,:]
y_test=label_y[3984:,:,:]"""
#x_train, x_test, y_train, y_test = train_test_split(vec_x, label_y, test_size=0.2, random_state=1)
#print x_train.shape[1:]
model = Sequential()
#print x_train.shape[1:]
model.add(LSTM(units=4,input_shape=(None,300), return_sequences=True))
model.add(Bidirectional(LSTM(units=4,return_sequences=True)))
model.add(LSTM(units=4,return_sequences=True))
model.add(LSTM(units=4,return_sequences=True))
model.add(TimeDistributed(Dense(3, activation='softmax')))
model.compile('rmsprop', 'categorical_crossentropy',metrics=['accuracy'])
#model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#model.fit(x_train, y_train, nb_epoch=100, validation_data=(x_test, y_test))
n_epochs = 10

for i in range(n_epochs):
    print("Training epoch {}".format(i))

    # bar = progressbar.ProgressBar(max_value=len(train_x))
    for j in range(len(vec_x)):
        #label = train_label[n_batch]
        label=y[j]
        # Make labels one hot
        #print(label)
        label = np.eye(3)[label][np.newaxis, :]
        # View each sentence as a batch
        sent = np.array(vec_x[j], dtype=np.float64)
        sent = sent[np.newaxis, :]
        #print(sent.shape,label.shape)
        if sent.shape[1] > 1 and sent.shape[1]==label.shape[1]:  # ignore 1 word sentences
            model.train_on_batch(sent,label)
#print(label.shape[1])
#loss,acc=model.evaluate()
sent = np.array(vec_x[4], dtype=np.float64)
sent = sent[np.newaxis, :]
#model.save('LSTM10.h5');
pred=model.predict_on_batch(sent)
pred=np.argmax(pred,-1)[0]

"""architecture=model.to_json()
with open('architecture.json','wt') as json_file:
    json_file.write(architecture)
model.save_weights('weights.h5')"""    
"""model.fit(x_train, y_train, nb_epoch=500, validation_data=(x_test, y_test))
model.save('LSTM1000.h5');
model.fit(x_train, y_train, nb_epoch=500, validation_data=(x_test, y_test))
model.save('LSTM1500.h5');
model.fit(x_train, y_train, nb_epoch=500, validation_data=(x_test, y_test))
model.save('LSTM2000.h5');
model.fit(x_train, y_train, nb_epoch=500, validation_data=(x_test, y_test))
model.save('LSTM2500.h5');
model.fit(x_train, y_train, nb_epoch=500, validation_data=(x_test, y_test))
model.save('LSTM3000.h5');
model.fit(x_train, y_train, nb_epoch=500, validation_data=(x_test, y_test))
model.save('LSTM3500.h5');
model.fit(x_train, y_train, nb_epoch=500, validation_data=(x_test, y_test))
model.save('LSTM4000.h5');
model.fit(x_train, y_train, nb_epoch=500, validation_data=(x_test, y_test))
model.save('LSTM4500.h5');
model.fit(x_train, y_train, nb_epoch=500, validation_data=(x_test, y_test))
model.save('LSTM5000.h5');"""
#predictions = model.predict(x_test)
#loss,accuracy=model.evaluate(x_test,y_test)
#mod = gensim.models.Word2Vec.load('word2vec.bin');
#[mod.most_similar([predictions[10][i]])[0] for i in range(15)]"""





