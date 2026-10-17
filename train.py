'''
Project Name  : dobot_ai
File Name     : train.py
File Encoding : UTF-8
Copyright © 2020 Afrel Co.,Ltd.
'''

import os, glob, random
import sys, time
from datetime import datetime
from enum import IntEnum, auto
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Kerasのモジュール群のインポート（1）
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import train_test_split

from common import *


def get_option():
    argparser = ArgumentParser()
    argparser.add_argument("-s", "--size", 
                            dest    = "pic_size",
                            type    = int,
                            default = PIC_SIZE,
                            help    = "Picture size when training." )
    argparser.add_argument("-e", "--epoch", 
                            dest    = "epoch",
                            type    = int,
                            default = EPOCH,
                            help    = "Number of training sessions." )
    argparser.add_argument("-b", "--batch", 
                            dest    = "batch",
                            type    = int,
                            default = BATCH,
                            help    = "Batch size" )
    argparser.add_argument("-r", "-R", "--retrain", 
                            dest    = "retrain_flag",
                            action  = "store_true",
                            help    = "Whether to Re-training." )
    argparser.add_argument("-d", "--directory", 
                            dest    = "load_dir",
                            type    = str,
                            default = DATA_DIR,
                            help    = "Directory to load pictures.")
    argparser.add_argument("-m", "--modelname", 
                            dest    = "modelname",
                            type    = str,
                            default = "model",
                            help    = "Output model name. (***.json, ***.weights.h5)" )
    return argparser.parse_args()


# 学習用データのロード（2）
def load_trainingData(outfile):
    '''
    学習用データ（教師データ）を読み込む
    '''
    image_list = []
    label_list = []
    if os.path.isfile(outfile) == False:
        for dir in os.listdir(DATA_DIR):
            if dir == ".DS_Store":
                continue
            if not(dir in LabelName):
                label_length = len(LabelName)
                LabelName.append(dir)
                setattr(LabelNumber, LabelName[label_length], label_length)
            dir1 = DATA_DIR + dir 
            print(dir1)

             # 配列label_listに正解ラベルを追加
            label = getattr(LabelNumber, dir)

            files = glob.glob(dir1 + "/*.png")
            random.shuffle(files)

            for i, f in enumerate(files):
                image = cv2.imread(f)
                image = cv2.resize(image, (PIC_SIZE, PIC_SIZE))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.asarray(image)
                image_list.append(image)
                label_list.append(label)
            print("%5d : Label[ %d ] %s\n" %(i, label, LabelName[label]))

        np.savez(outfile, x=image_list, y=label_list)
        print("\nTotal %d images  --> Saved : %s" % (len(image_list), outfile))
    else:
        for dir in os.listdir(DATA_DIR):
            if dir == ".DS_Store":
                continue
            elif dir == OUTFILE_NAME:
                continue
            dir1 = DATA_DIR + dir 
            print(dir1)
            if not(dir in LabelName):
                label_length = len(LabelName)
                LabelName.append(dir)
                setattr(LabelNumber, LabelName[label_length], label_length)
                print(getattr(LabelNumber, LabelName[label_length]), " : ", str(LabelName[label_length]), "\n")

    photos = np.load(outfile)
    image_list = photos["x"]
    label_list = photos["y"]

    print("\n - - - label check - - - \n")
    cnt = 0
    for m in LabelName:
        setattr(LabelNumber, LabelName[cnt], cnt)
        print(getattr(LabelNumber, LabelName[cnt]), " : ", str(LabelName[cnt]))
        cnt += 1

    return image_list, label_list


# モデルの構築（3）
def definition_model(input_shape, output_classes):
    '''
    モデルを生成してニューラルネットを構築
    '''
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding=("same")))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(output_classes))
    model.add(Activation("softmax"))

    return model


# 学習の方法の設定（4）
def compile_model(model):
    '''
    オプティマイザーを設定して、モデルをコンパイル
    '''
    # 学習スケジュールを設定
    learning_rate_schedule = optimizers.schedules.ExponentialDecay( initial_learning_rate=0.001,  decay_steps=100000,  decay_rate=0.96,  staircase=True )
    # オプティマイザにRMSpropを使用
    #opt = optimizers.RMSprop(lr = 0.0001, decay=1e-6)
    opt = optimizers.RMSprop(learning_rate =learning_rate_schedule)

    # モデルをコンパイル
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # モデル構成を表示
    model.summary()


# 学習の実行（5）
if __name__ == '__main__':
    args = get_option()
    PIC_SIZE = args.pic_size
    EPOCH    = args.epoch
    BATCH    = args.batch
    DATA_DIR = args.load_dir
    
    # 学習データの保存ファイル名
    outfile = DATA_DIR + OUTFILE_NAME

    if args.retrain_flag == False and os.path.isfile(MODEL_DIR + args.modelname + ".json") == True:
        args.retrain_flag = False
    else:
        args.retrain_flag = True

    # 学習用データのロード（5-1）
    print("\n - - - training data load - - - \n")
    image_list, label_list = load_trainingData(outfile)

    # データの前処理（5-2）
    print("\n - - - - - - - - - - - - - -\n")
    image_list = image_list.reshape(-1, PIC_SIZE, PIC_SIZE, 3)
    image_list = image_list.astype("float32") / 255.
    print("image_list -> ", image_list.shape)

    # ラベルデータをone-hotベクトルに直す
    # 0 -> [1,0,0], 1 -> [0,1,0] 
    label_list = to_categorical(label_list.astype("int32"), len(LabelName))
    print("label_list -> ", label_list.shape)

    # 学習用とテスト用に分ける（5-3）
    x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, train_size = 0.9)
    print("x_train    -> ", x_train.shape)
    print("x_test     -> ", x_test.shape)
    print("y_train    -> ", y_train.shape)
    print("y_test     -> ", y_test.shape)
    print("\n - - - - - - - - - - - - - -\n")


    # 再学習せず、テストのみ（モデルデータ（jsonファイル）があり、-rオプションを付けていない場合）
    if args.retrain_flag == False:
        print("\n - - - model data loading - - - \n")
        # モデルを読み込み （5-4）
        model  = model_from_json(open(MODEL_DIR + args.modelname + ".json").read())
        # パラメーターを読み込み
        model.load_weights(MODEL_DIR + args.modelname + ".weights.h5")
        # オプティマイザーを指定して、モデルをコンパイル
        compile_model(model)

        print("\n - - - test - - - \n")
        # テストを実行（5-5）
        score = model.evaluate(x_test, y_test, verbose = 1)
        print("\n正解率 = %5.2f %%, loss = %.10f" % (score[1] * 100., score[0]))
        print("score -> ", score)


    # はじめて学習する場合（モデルデータがない）
    # または、再学習する場合（-rオプションを付けている）
    else:
        print("\n - - - declare model - - - \n")
        # モデルを定義する（5-6）
        model = definition_model(x_train.shape[1:], len(LabelName))
        # オプティマイザーを指定して、モデルをコンパイル（5-7）
        compile_model(model)

        print("\n - - - training start - - - \n")
        # 学習を実行（5-8）
        history = model.fit(x = x_train, y = y_train, epochs = EPOCH, batch_size = BATCH, verbose = 1, validation_split = 0.1)

        print("\n - - - test - - - \n")
        # テストを実行（5-9）
        score = model.evaluate(x_test, y_test, verbose = 1)
        print("\n正解率 = %5.2f %%, loss = %.10f" % (score[1] * 100., score[0]))
        print("score -> ", score)

        # モデルを保存（5-10）
        model_json = model.to_json()
        open(os.path.join(MODEL_DIR, args.modelname+".json"), "w").write(model_json)
        model.save_weights(os.path.join(MODEL_DIR, args.modelname+".weights.h5"))

        print("\n ---> Output model ( " + args.modelname + ".json, " + args.modelname + ".weights.h5 )")


        # 学習結果を保存、表示（5-11）
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

        # historyの表示（loss）
        axL.plot(history.history['loss'],     label="loss",     ls="-")
        axL.plot(history.history['val_loss'], label="val_loss", ls="-")
        axL.set_title('model loss')
        axL.set_xlabel('epoch')
        axL.set_ylabel('loss')
        axL.legend(loc = 'best')

        # historyの表示（accuracy）
        axR.plot(history.history['accuracy'],     label="accuracy",     ls="-")
        axR.plot(history.history['val_accuracy'], label="val_accuracy", ls="-")
        axR.set_title('model accuracy')
        axR.set_xlabel('epoch')
        axR.set_ylabel('accuracy')
        axR.legend(loc = 'best')

        nowtime = datetime.now().strftime("_%y%m%d_%H%M%S_")
        filepath = HISTORY_DIR + "hist" + args.modelname + nowtime + ".png"

        if os.path.isdir(HISTORY_DIR) == False:
            if os.makedirs(HISTORY_DIR) == False:
                print("\n No such directory. \""+ HISTORY_DIR + "\"")
                os.sys.exit()
            print("\n Make directory. \""+ HISTORY_DIR + "\"")

        fig.savefig(filepath)
        plt.show()
        plt.close()

    # kerasの終了処理（5-12）
    clear_session()
