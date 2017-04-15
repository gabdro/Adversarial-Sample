import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import matplotlib.pyplot as plt
import cv2
import pandas as pd
import MODEL  #CIFAR10_cifer()
from skimage import io

#モデルの読み込み
model = MODEL.L.Classifier(MODEL.CIFAR10_cifer())
serializers.load_npz("model_final",model)

def Print_Label(num):
    label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    #print("Label->{}".format(label[num]))
    return label[num]

#ニューロンを一覧で表示する
def MELMEL(data,num=10):
    #print("number : neuron")
    dict = {}
    for i in range(num):
        dict.update({Print_Label(i):data[0][i]*100})
    #print(Print_Label(i),":",int(data[0][i]*100))
    return dict

#分類器を用いた画像の予想
#この関数では画像の各ニューロン値及び「予想したラベル」を返す
def predict(pic):
    t = chainer.Variable(np.asarray([[]], dtype=np.int32), volatile="on")
    pic = chainer.Variable(np.asarray([pic],dtype=np.float32),volatile="on")
    #print(model.predictor(IM,t).data)
    predict_label = F.softmax(model.predictor(pic,t).data).data.argmax()
    #print(PrintLabel(res))
    return model.predictor(pic,t).data , predict_label

##Create Adversarial Exmaple
## 本関数ではAdversarial ExmapleのためのNoise Tableを返す関数
#im' 画像 #target_true'正しいラベル #target_dummy'偽装したいラベル(数字)
def AE(im,target_true,target_dummy):
    S=32 #画像サイズ
    r=0.001 #加えるノイズ量
    #事前に受け取ったノイズを加える前の画像に対する分類器の出力値を保存
    RES,I = predict(im.reshape(3,S,S)) #RESは各ニューロン値 Iは最も正しいと判断したラベル(本実験では使わないかもしれない)
    
    noise_table = np.zeros(3*S*S)
    
    view = im.ravel()
    diff_res =RES[0][target_dummy] - RES[0][target_true]
    for k in range(3*S*S):
        backup = view[k]
        view[k] += r
        ans , I = predict(view.reshape(3,S,S))
        if ans[0][target_dummy] - ans[0][target_true] < diff_res:
            noise_table[k] = -1
        else:
            noise_table[k] = +1
        view[k] = backup
    
    return noise_table

##main
train,test = datasets.get_cifar10()
CIFAR_TEST_LENGTH = len(test)

#CIFAR-10のテストデータからAEを生成してcsv形式で保存する機構
df_label = pd.DataFrame(columns=["pos","adv","predict"])
epsilon = 0.01
all_label = [0,1,2,3,4,5,6,7,8,9]
count = 0
for i in range(CIFAR_TEST_LENGTH):
    pos = test._datasets[1][i]
    for adv_class in all_label:
        if adv_class == test._datasets[1][i]:
            break
        else:
            N_table = AE(test._datasets[0][i],pos,adv_class)
            N_table = N_table.reshape(3,32,32)
            adv_pic = test._datasets[0][i] + epsilon*N_table
            adv_pic[adv_pic>1] , adv_pic[adv_pic < 0] =1,0 #1を超えた値を1にキャスト,マイナスになったら0にキャスト
            R1,R2 = predict(adv_pic) #R1が予想した各ニューロン値 #R2が予想したラベル
            #np.save(str(i)+"_"+str(adv_class)+".npy",adv_pic)
            np.save("adv_data/"+str(count)+".npy",adv_pic)
            df_label.loc[count] = [pos,adv_class,R2]
            count += 1
df_label.to_csv("label_index.csv")