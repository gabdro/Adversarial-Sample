import chainer
import chainer.functions as F

#gradient(全ラベルと予測値),第一予測ラベル
def predict(pic):
    t = chainer.Variable(np.asarray([[]], dtype=np.int32), volatile="on")
    pic = chainer.Variable(np.asarray([pic],dtype=np.float32),volatile="on")
    predict_label = F.softmax(model.predictor(pic,t).data).data.argmax()
    
    return model.predictor(pic,t).data[0] , predict_label