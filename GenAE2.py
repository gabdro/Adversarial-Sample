import numpy as np
import chainer
import MODEL  #CIFAR10_cifer()
import h5py

from concurrent.futures import ThreadPoolExecutor
import time

import pandas as pd

dummy_label = chainer.Variable(np.empty((1,0)), volatile="on")

def predict(model, inputs):
    x = chainer.Variable(inputs, volatile="on")
    return model.predictor(x, dummy_label).data

def calc_AE(model, image, label, n_target, r=0.001, n_threads=32):
    # np.newaxis は次元を増やす
    score0 = predict(model, image[np.newaxis,:,:,:]).T
    # 最初から１ピクセルずつ変更した画像を作ってしまう（メモリが必要）
    size = image.size
    images = np.tile(image, (size, 1, 1, 1))
    view = images.reshape(size, size)
    for k in range(size):
        view[k,k] += r
    # 分割
    n = n_threads
    args = [images[size*i//n:size*(i+1)//n] for i in range(n-1)]
    args.append(images[size*(n-1)//n:size])
    # 並列実行
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        score = executor.map(lambda x: predict(model, x), args)
    # まとめ
    score = np.vstack([s for s in score]).T
    inc = (score - score[label]) - (score0 - score0[label])
    return inc.reshape(n_target, *image.shape)

def create_AE_file(filename, model, images, labels, r=0.001, n_threads=32):
    CIFAR_TEST_LENGTH = images.shape[0]
    C, H, W = images[0].shape
    N_TARGET = np.unique(labels).size
    with h5py.File(filename) as hdf5:
        # 画像ごとループ
        for img_id in range(CIFAR_TEST_LENGTH):
            # 保存済みならスキップ
            str_id = str(img_id)
            if str_id in hdf5:
                if 'finish' in hdf5[str_id].attrs:
                    continue
                del hdf5[str_id]

            # noise_table 作成
            # N_TARGET, ピクセルごとのscore増加量
            start = time.time()
            image, label = images[img_id], labels[img_id]
            noise_tables = calc_AE(model, image, label, N_TARGET, r, n_threads)
            print('{} sec'.format(time.time() - start))

            # グループを作って保存
            g1 = hdf5.create_group(str_id)
            g1.create_dataset('original', data=image)
            g1.attrs['id'] = img_id
            g1.attrs['label'] = label
            g2 = g1.create_group('inc')
            for target in range(N_TARGET):
                if target != label:
                    ds = g2.create_dataset(str(target), data=noise_tables[target])
                    ds.attrs['target'] = target
            hdf5.flush()
            # img_id が保存されたことを保証する
            g1.attrs['finish'] = 1
            hdf5.flush()

def evaluate_AE(filename, model, epsilon, n_threads=32):
    def local_eval(g1):
        # 元データ読み込み
        image = g1['original'][:]
        img_id = g1.attrs['id']
        label = g1.attrs['label']
        # ノイズテーブル用グループ　
        g2 = g1['inc']
        # AE 画像 (epsilon) 用グループ
        str_eps = str(epsilon)
        if str_eps in g1:
            del g1[str_eps]
        g3 = g1.create_group(str_eps)
        # 評価
        out = []
        for j in g2:
            # ノイズテーブル (inc)
            data = g2[j]
            target = data.attrs['target']
            inc = data[:]
            # ノイズ作成
            noise = np.zeros_like(inc, 'i1')
            noise[inc>0] = +1
            noise[inc<0] = -1
            # AE 画像作成
            adv_image = image.copy()
            adv_image += epsilon * noise
            adv_image[adv_image>1] = 1
            adv_image[adv_image<0] = 0
            # 予測
            pred = np.argmax(predict(model, adv_image[np.newaxis,:,:,:]))
            # hdf5 書き込み
            ds = g3.create_dataset(j, data=adv_image)
            ds.attrs['predict'] = pred
            # out に書き込み
            out.append([img_id, label, target, pred])
        return out
    with h5py.File(filename) as hdf5:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            out = np.vstack(executor.map(local_eval, [hdf5[i] for i in hdf5]))
    return pd.DataFrame(out, columns=["ids", "pos", "adv", "predict"])

def main():
    # モデルの読み込み
    model = MODEL.L.Classifier(MODEL.CIFAR10_cifer())
    chainer.serializers.load_npz("model_final", model)
    # データの読み込み
    images, labels = chainer.datasets.get_cifar10()[1]._datasets
    # table 作成
    create_AE_file('AE.h5', model, images, labels, r=0.001, n_threads=32)
    # 評価
    df = evaluate_AE('AE.h5', model, epsilon=0.01, n_threads=32)
    # 書き出し
    df.to_csv('AE.csv')

if __name__ == '__main__':
    main()
