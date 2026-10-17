'''
Project Name  : dobot_ai
File Name     : shoot_trainingData.py
File Encoding : UTF-8
Copyright © 2020 Afrel Co.,Ltd.
'''

import os
import sys, time
from datetime import datetime
from argparse import ArgumentParser

import cv2
import numpy as np

# WEBカメラ制御用モジュール、共通設定のインポート（1）
import cameraSetting as camset
from common import *


# Data Augmentation（2）
def save_image(img_src, capstr, cnt = 0):
    '''
    補正をかけた画像を保存する
    '''
    save_cnt = cnt

    # リストに格納された長方形を画像(png)データで保存する
    cv2.imwrite(capstr + ".png", img_src)
    save_cnt += 1
    print(capstr + ".png")

    # モザイク
    # resize_img = add_mosaic(img_src)
    # cv2.imwrite(capstr + "_mosaic.png", resize_img)
    # save_cnt += 1
    # print(capstr + "_mosaic.png")

    # ぼかし
    # gauss_img = add_gaussianBlur(img_src)
    # cv2.imwrite(capstr + "_gauss.png", gauss_img)
    # save_cnt += 1
    # print(capstr + "_gauss.png")

    # 高コントラス調整
    high_cont_img = adjust_to_highContrast(img_src)
    cv2.imwrite(capstr + "_high_cont.png", high_cont_img)
    save_cnt += 1
    print(capstr + "_high_cont.png")

    # 低コントラスト調整
    low_cont_img = adjust_to_lowContrast(img_src)
    cv2.imwrite(capstr + "_low_cont.png", low_cont_img)
    save_cnt += 1
    print(capstr + "_low_cont.png")

    # ガンマ調整
    low_gamma_1_img = adjust_Gamma(img_src, gamma = 0.5)
    cv2.imwrite(capstr + "_low_gamma_1.png", low_gamma_1_img)
    save_cnt += 1
    print(capstr + "_low_gamma_1.png")

    low_gamma_2_img = adjust_Gamma(img_src, gamma = 0.75)
    cv2.imwrite(capstr + "_low_gamma_2.png", low_gamma_2_img)
    save_cnt += 1
    print(capstr + "_low_gamma_2.png")

    high_gamma_1_img = adjust_Gamma(img_src, gamma = 1.5)
    cv2.imwrite(capstr + "_high_gamma_1.png", high_gamma_1_img)
    save_cnt += 1
    print(capstr + "_high_gamma_1.png")

    high_gamma_2_img = adjust_Gamma(img_src, gamma = 2.0)
    cv2.imwrite(capstr + "_high_gamma_2.png", high_gamma_2_img)
    save_cnt += 1
    print(capstr + "_high_gamma_2.png")

    # ガウス分布に基づくノイズ
    gauss_noise_img = add_gaussianNoise(img_src)
    cv2.imwrite(capstr + "_gauss_noise.png", gauss_noise_img)
    save_cnt += 1
    print(capstr + "_gauss_noise.png")

    '''
    # Salt&Pepperノイズ
    sp_noise_img = add_saltAndPepperNoise(img_src)
    cv2.imwrite(capstr + "_sp_noise.png", sp_noise_img)
    save_cnt += 1
    print(capstr + "_sp_noise.png")

    return save_cnt
    '''

def add_mosaic(img_src):
    '''
    モザイク処理
    '''
    # 1/25に縮小
    image = cv2.resize(img_src, (int(img_src.shape[1]/5), int(img_src.shape[0]/5)))
    # 元サイズに拡大
    image = cv2.resize(image, (int(img_src.shape[1]), int(img_src.shape[0])))
    return image

def add_gaussianBlur(img_src):
    '''
    ぼかし処理
    '''
    image = cv2.GaussianBlur(img_src, (15, 15), 0)
    return image

def add_gaussianNoise(img_src):
    '''
    ガウス分布に基づくノイズ処理
    '''
    row, col, ch = img_src.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    image = img_src + gauss
    return image

def add_saltAndPepperNoise(img_src):
    '''
    Salt&Pepperノイズ処理
    '''
    row, col, ch = img_src.shape
    s_vs_p = 0.5
    amount = 0.004
    image = img_src.copy()

    # 塩モード
    num_salt = np.ceil(amount * img_src.size * s_vs_p)
    coords = [np.random.randint(0, i-1 , int(num_salt)) for i in img_src.shape]
    image[coords[:-1]] = (255,255,255)
    # 胡椒モード
    num_pepper = np.ceil(amount* img_src.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in img_src.shape]
    image[coords[:-1]] = (0,0,0)

    return image

def adjust_to_highContrast(img_src):
    '''
    ハイコントラスト調整
    '''
    # ルックアップテーブルの生成
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table
    LUT_HC = np.arange(256, dtype = 'uint8' )

    # ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0
    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table
    for i in range(max_table, 255):
        LUT_HC[i] = 255

    image = cv2.LUT(img_src, LUT_HC)
    return image

def adjust_to_lowContrast(img_src):
    '''
    ローコントラスト調整
    '''
    # ルックアップテーブルの生成
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table
    LUT_LC = np.arange(256, dtype = 'uint8' )

    # ローコントラストLUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255

    image = cv2.LUT(img_src, LUT_LC)
    return image

def adjust_Gamma(img_src, gamma):
    '''
    ガンマ調整
    '''
    # ガンマ変換ルックアップテーブル
    LUT_G = np.arange(256, dtype = 'uint8' )
    for i in range(256):
        LUT_G[i] = 255 * pow(float(i) / 255, 1.0 / gamma)

    image = cv2.LUT(img_src, LUT_G)
    return image

# 画像のサイズ調整（3）
def rect_preprocess(img):
    '''
    切り取った矩形の長辺に合わせて短辺を伸ばす
    伸ばされた部分は、黒色＝RGB[0, 0, 0]で塗りつぶす
    '''
    # 長辺のサイズを取得
    h, w, c = img.shape
    longest_edge = max(h, w)
    top = bottom = left = right = 0

    # 横のほうが大きい場合
    if h < longest_edge:
        diff_h = longest_edge - h
        top = diff_h // 2
        bottom = diff_h - top
    # 縦のほうが大きい場合
    elif w < longest_edge:
        diff_w = longest_edge - w
        left = diff_w // 2
        right = diff_w - left
    else:
        pass
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img


def get_option():
    argparser = ArgumentParser()
    argparser.add_argument("-d", "--directory",
                            dest    = "save_dir",
                            type    = str,
                            default = DATA_DIR,
                            help    = "Directory for saving pictures.")
    argparser.add_argument("-f", "--filename",
                            dest    = "filename",
                            type    = str,
                            default = CAPTURE_NAME,
                            help    = "Capture file name.")
    argparser.add_argument("--min",
                            dest    = "min_area_size",
                            type    = int,
                            default = MIN_AREA_SIZE,
                            help    = "Minimum area size." )
    argparser.add_argument("--max",
                            dest    = "max_area_size",
                            type    = int,
                            default = MAX_AREA_SIZE,
                            help    = "Maximum area size." )
    return argparser.parse_args()


# 教師データの作成（4）
if __name__ == '__main__':
    args = get_option()
    DATA_DIR      = args.save_dir
    CAPTURE_NAME  = args.filename
    MIN_AREA_SIZE = args.min_area_size
    MAX_AREA_SIZE = args.max_area_size

    if os.path.isdir(DATA_DIR) == False:
        if os.makedirs(DATA_DIR) == False:
            print("\n No such directory. \""+ DATA_DIR + "\"")
            os.sys.exit()
        print("\n Make directory. \""+ DATA_DIR + "\"")


    print("\n The directory to save is " + DATA_DIR)
    print(" The filename  to save is " + DATA_DIR + CAPTURE_NAME + "_YYMMDD_HHMMSS_X" + ".png")

    # VideoCaptureのインスタンスを作成する（4-1）
    cap = cv2.VideoCapture(2)

    print("\n - - - - - - - - - - ")
    # camset.camera_set(cv2, cap, gain = **調整した値**, exposure = **調整した値**.)
    camset.camera_get(cv2, cap)
    print(" - - - - - - - - - - \n")

    print()
    print(" Press [ S ] key to save image.")
    print(" Press [ A ] key to save the corrected image.")
    print(" Press [ R ] key to save corrected and rotated by 45 degree images.")
    print()
    print(" Press [ C ] key to Gain, Exposure setting.")
    print(" Press [ESC] key to exit.")
    print()

    while True:
        # VideoCaptureから1フレーム読み込む（4-2）
        ret, frame = cap.read()
        ret, edframe = cap.read()
        # 加工なし画像を表示する
        cv2.imshow('Raw Frame', frame)

        # グレースケールに変換（4-3）
        gray = cv2.cvtColor(edframe, cv2.COLOR_BGR2GRAY)
        # 1/4サイズに縮小
        gray_s = cv2.resize(gray, (int(gray.shape[1]/2), int(gray.shape[0]/2)))
        # グレースケール画像を表示する
        cv2.imshow('Gray Frame', gray_s)

        # ２値化（4-4）
        retval, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 1/4サイズに縮小
        bw_s = cv2.resize(bw, (int(gray.shape[1]/2), int(gray.shape[0]/2)))
        # ２値化画像を表示する
        cv2.imshow('Binary Frame', bw_s)


        # 輪郭を抽出（4-5）
        #   contours : [領域][Point No][0][x=0, y=1]
        #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
        #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cutframe_array = []
        # 各輪郭に対する処理
        for contour in contours:
            # 輪郭の領域を計算（4-6）
            area = cv2.contourArea(contour)

            # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外（4-7）
            if area < MIN_AREA_SIZE or MAX_AREA_SIZE < area:
                continue

            # フレーム画像から対象物を切り出す（4-8）
            # 回転を考慮した外接矩形を取得する
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(edframe, [box], 0, draw_red, 2)

            center, size, angle = rect
            center = tuple(map(int, center))  # float -> int
            size   = tuple(map(int, size))    # float -> int

            # 回転行列を取得する
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            h, w = frame.shape[:2]

            # 切り出す
            rotated = cv2.warpAffine(frame, rot_mat, (w, h))
            cropped = cv2.getRectSubPix(rotated, size, center)
            cutframe_array.append(cropped)

            # 輪郭に外接する長方形を取得する
            x, y, width, height = cv2.boundingRect(contour)
            # 輪郭に外接する長方形を描画する
            cv2.rectangle(edframe, (x, y), (x+width, y+height), draw_white)
            # 長方形の各頂点を描画する
            cv2.drawMarker(edframe, (box[0][0], box[0][1]), draw_green,  cv2.MARKER_CROSS, thickness = 1) # 一番下の座標
            cv2.drawMarker(edframe, (box[1][0], box[1][1]), draw_yellow, cv2.MARKER_CROSS, thickness = 1) # 以下、時計回りに座標格納されている
            cv2.drawMarker(edframe, (box[2][0], box[2][1]), draw_blue,   cv2.MARKER_CROSS, thickness = 1) #
            cv2.drawMarker(edframe, (box[3][0], box[3][1]), draw_red,    cv2.MARKER_CROSS, thickness = 1) #

            # 輪郭データを浮動小数点型の配列に格納
            X = np.array(contour, dtype=np.float64).reshape((contour.shape[0], contour.shape[2]))
            # PCA（１次元）
            mean, eigenvectors = cv2.PCACompute(X, mean=np.array([], dtype=np.float64), maxComponents=1)
            # 中心を描画
            mp_x = int(mean[0][0])
            mp_y = int(mean[0][1])
            cv2.drawMarker(edframe, (mp_x, mp_y), draw_black, cv2.MARKER_TILTED_CROSS, thickness = 1)

            # 情報を描画
            label = " Mid : (" + str(mp_x) + ", " + str(mp_y) + ")"
            cv2.putText(edframe, label, (x+width, y+10), font, FONT_SIZE, draw_green, FONT_WIDTH, cv2.LINE_AA)
            label = " Area: " + str(area)
            cv2.putText(edframe, label, (x+width, y+30), font, FONT_SIZE, draw_white, FONT_WIDTH, cv2.LINE_AA)

        # 描画した画像を表示
        cv2.imshow('Edited Frame', edframe)

        # キー入力を1ms待つ
        k = cv2.waitKey(1)

        # 「ESC（27）」キーを押す
        # プログラムを終了する
        if k == 27:
            break

        # 「C」キーを押す
        # WEBカメラのゲイン値、露出の値を調整する
        elif k == ord('c'):
            g = input("gain     : ")
            e = input("exposure : ")
            print("\n - - - - - - - - - - ")
            camset.camera_set(cv2, cap, gain = float(g), exposure = float(e))
            camset.camera_get(cv2, cap)
            print(" - - - - - - - - - - \n")

       # 画像の保存（4-9）
       # 「S」キーを押す
       # そのまま切り取って画像を保存する
        elif k == ord('s'):
            w_result = True
            save_cnt = 0
            nowtime = datetime.now().strftime("_%y%m%d_%H%M%S_")
            for cnt in range(0, len(cutframe_array)):
                # リストに格納された矩形を長辺に合わせてサイズ調整する
                img_src = rect_preprocess(cutframe_array[cnt])
                # サイズ調整した正方形を画像(png)データで保存する
                capstr = DATA_DIR + CAPTURE_NAME + nowtime + str(cnt) + ".png"
                cv2.imwrite(capstr, img_src)
                save_cnt += 1
                print(capstr)
            print(" - - - - - - - - - - " + str(save_cnt) + " images saved\n")


        # 「A」キーを押す
        # 補正を加えた画像を保存する
        elif k == ord('a'):
            w_result = True
            save_cnt = 0
            nowtime = datetime.now().strftime("_%y%m%d_%H%M%S_")
            for cnt in range(0, len(cutframe_array)):
                # 取得した矩形を長辺に合わせてサイズ調整する
                img_src = rect_preprocess(cutframe_array[cnt])
                capstr = DATA_DIR + CAPTURE_NAME + nowtime + str(cnt)
                # サイズ調整した正方形に補正を加えて保存する
                save_cnt = save_image(img_src, capstr, save_cnt)

            print(" - - - - - - - - - - " + str(save_cnt) + " images saved\n")


        # 「R」キーを押す
        # 画像を回転させた上に補正を加えた画像を保存する
        elif k == ord('r'):
            w_result = True
            save_cnt = 0
            nowtime = datetime.now().strftime("_%y%m%d_%H%M%S_")
            for cnt in range(0, len(cutframe_array)):
                # 取得した矩形を長辺に合わせてサイズ調整する
                img_src = rect_preprocess(cutframe_array[cnt])
                # 画像の中心位置
                center = tuple(np.array([img_src.shape[1] * 0.5, img_src.shape[0] * 0.5]))
                # 画像サイズの取得(横, 縦)
                size = tuple(np.array([img_src.shape[1], img_src.shape[0]]))

                # リストに格納された長方形を画像(png)データで保存
                # 回転（0°, 90°, 180°, 270°）して、変換処理した画像を保存
                for j in range(0, 4):
                    rot = 90 * j
                    # 回転変換行列の算出
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle=rot, scale=1.0)
                    # アフィン変換
                    rot_img = cv2.warpAffine(img_src, rotation_matrix, size, flags=cv2.INTER_CUBIC)

                    capstr = DATA_DIR + CAPTURE_NAME + nowtime + str(cnt) + "_rot" + str(rot)
                    save_cnt = save_image(rot_img, capstr, save_cnt)

            print(" - - - - - - - - - - " + str(save_cnt) + " images saved\n")


    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    cv2.destroyAllWindows()
