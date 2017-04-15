import matplot.pyplot as plt
import numpy as np
import chainer

def GenAS(pic,target_true,target_adv):
    plt.imshow(np.rollaxis(pic,0,3)) #original 表示
  
    color,row,col = pic.shape #3,32,32
    BitMap = np.zeros(color*row*col)
    #print(pic.value())
    view = pic.ravel()
  
    softmax_list = []
    r=0.01 #摂動の振幅
  
    limit = 112 #112より大きいと人間に検出
    for i in range(0,limit):
        res, predict_label = predict_P(pic)
        diff_res = res[target_adv] - res[target_true]
        pred_numList = []
        pred_max = 0
        for k in range(color*row*col):
                backup = view[k]
                view[k] += r
                ans , ans_label = predict_P(view.reshape(color,row,col))
                L = diff_res + (ans[target_adv] - ans[target_true])
      
                # 予測(L)を全てリストに追加する.
                pred_numList.append(L)
      
                 # 予測値の最大値を求めるため,差分(L)の絶対値 と 既存の(L)を比較する.
                # 2ループ目以降,既に確定して摂動を加えた箇所は無視する.
                if abs(L) >= pred_max and BitMap[k] == 0:
                    pred_max = abs(L) #値の絶対値の最大値を記録
                    max_k = k #値の絶対値の最大値の番号を記録
            
                view[k] = backup #origin
  
        #CIFAR10では[+2,-2] , MNISTでは[+1,-1]
        if pred_numList[max_k] > 0:
            view[max_k] += 2
        else:
            view[max_k] += -2
  
        # Adversarial Sampleになっているか確認.
        # 予測結果が正しいラベルと異なっていれば作成したとする.
        p = predict_P(view.reshape(color,row,col))[1]
        if target_true != p:
            print("This pic pred :",p)
            print("success adversarial sample!")
            return view.reshape(color,row,col)
    print("Fail")
    return view.reshape(color,row,col)