import os, sys
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras import optimizers

classes = ['chino', 'cocoa', 'chiya', 'rize', 'syaro']
nb_classes = len(classes)
img_width, img_height = 150, 150

result_dir = 'results'

# このディレクトリにテストしたい画像を格納しておく
test_data_dir = 'dataset/test'

def model_load():
    # VGG16, FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC層の作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # VGG16とFC層を結合してモデルを作成
    model = Model(input=vgg16.input, output=top_model(vgg16.output))

    # 学習済みの重みをロード
    model.load_weights(os.path.join(result_dir, 'finetuning.h5'))

    # 多クラス分類を指定
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

    return model


if __name__ == '__main__':

    # モデルのロード
    model = model_load()

    # テスト用画像取得
    test_imagelist = os.listdir(test_data_dir)

    for test_image in test_imagelist:
        filename = os.path.join(test_data_dir, test_image)
        img = image.load_img(filename, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # 学習時に正規化してるので、ここでも正規化
        x = x / 255
        pred = model.predict(x)[0]

        # 予測確率が高いトップを出力
        # 今回は最も似ているクラスのみ出力したいので1にしているが、上位n個を表示させることも可能。
        top = 1
        top_indices = pred.argsort()[-top:][::-1]
        result = [(classes[i], pred[i]) for i in top_indices]
        print('file name is', test_image)
        print(result)
        print('=======================================')
