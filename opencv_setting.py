'''
Project Name  : dobot_colorClassifier
File Name     : opencv_setting.py
File Encoding : UTF-8
Copyright © 2020 Afrel Co.,Ltd.
'''

import cv2
import numpy as np
from enum import IntEnum, auto

# DOBOT、WEBカメラ制御用モジュールのインポート(1)
import myDobotModule as dobot
import cameraSetting as camset
from common import *

# 各色(赤、青、緑、黄)に整数値（0,1,2,3）を割り当てる
class Color(IntEnum):
    RED    = auto()
    BLUE   = auto()
    GREEN  = auto()
    YELLOW = auto()

# 領域抽出用の最小／最大サイズ
MIN_AREA_SIZE = 50
MAX_AREA_SIZE = 800

# 測定マットのマーカーの座標
# RED
dobot_red    = [150, 0]
# BLUE
dobot_blue   = [300, 0]
# YELLOW
dobot_yellow = [150, -100]
# GREEN
dobot_green  = [300, -100]


#指定した色の領域の中心点を取得する（2）
def find_specific_color(color_name, frame, edframe, low, high):
    '''
    指定された色空間の範囲から色を抽出し、輪郭から領域、矩形を取得する
    '''
    # HSV色空間に変換（2-1）
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 色を抽出する（2-2）
    ex_img = cv2.inRange(hsv, low, high)
    # 1/4サイズに縮小して表示
    ex_img_s = cv2.resize(ex_img, (int(ex_img.shape[1]/2), int(ex_img.shape[0]/2)))
    cv2.imshow(color_name, ex_img_s)

    # 輪郭抽出（2-3）
    contours, hierarchy = cv2.findContours(ex_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mp_x = mp_y = None
    for i, contour in enumerate(contours):
        # ノイズの除去（2-4）
        # 輪郭の領域を計算
        area = cv2.contourArea(contour)
        # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
        if area < MIN_AREA_SIZE or MAX_AREA_SIZE < area:
            continue

        # 輪郭データの中心点を計算（2-5）
        # 輪郭データを浮動小数点型の配列に格納
        X = np.array(contour, dtype=np.float64).reshape((contour.shape[0], contour.shape[2]))
        # PCA（１次元）
        mean, eigenvectors = cv2.PCACompute(X, mean=np.array([], dtype=np.float64), maxComponents=1)

        # フレーム画像に描画（2-6）
        # 輪郭に外接する長方形を取得する
        x, y, width, height = cv2.boundingRect(contour)
        # 長方形を描画する
        cv2.rectangle(edframe, (x, y), (x+width, y+height), draw_red, thickness=1)

        # 中心を描画
        mp_x = int(mean[0][0])
        mp_y = int(mean[0][1])
        cv2.drawMarker(edframe, (mp_x, mp_y), draw_red, cv2.MARKER_TILTED_CROSS, thickness = 1)

        # 情報を描画
        label = " Mid : (" + str(mp_x) + ", " + str(mp_y) + ")"
        cv2.putText(edframe, label, (x+width, y+10), font, FONT_SIZE, draw_green, FONT_WIDTH, cv2.LINE_AA)
        label = " Area: " + str(area)
        cv2.putText(edframe, label, (x+width, y+30), font, FONT_SIZE, draw_green, FONT_WIDTH, cv2.LINE_AA)

    # 輪郭の中心点を返却（2-7）
    if mp_x == None and mp_y == None: return None
    return (mp_x, mp_y)


# 変換行列を作成する（3）
if __name__ == '__main__':
    # VideoCaptureのインスタンスを作成する（3-1）
    cap = cv2.VideoCapture(1)

    print("\n - - - - - - - - - - ")
    # camset.camera_set(cv2, cap, gain = **調整した値**, exposure = **調整した値**.)
    camset.camera_get(cv2, cap)
    print(" - - - - - - - - - - \n")

    print()
    print(" Press [ S ] key to get Transformation matrix.")
    print()
    print(" Press [ C ] key to Gain, Exposure setting.")
    print(" Press [ESC] key to exit.")
    print()


    while True:
        # VideoCaptureから1フレーム読み込む（3-2）
        ret, frame = cap.read()
        ret, edframe = cap.read()
        # 加工なし画像を表示する
        cv2.imshow('Raw Frame', frame)

        # マーカー座標（カメラ座標系）を取得する（3-3）
        # RED
        pos_red = find_specific_color("RED", frame, edframe, RED_LOW_COLOR, RED_HIGH_COLOR)
        # BLUE
        pos_blue = find_specific_color("BLUE", frame, edframe, BLUE_LOW_COLOR, BLUE_HIGH_COLOR)
        # YELLOW
        pos_yellow = find_specific_color("YELLOW", frame, edframe, YELLOW_LOW_COLOR, YELLOW_HIGH_COLOR)
        # GREEN
        pos_green = find_specific_color("GREEN", frame, edframe, GREEN_LOW_COLOR, GREEN_HIGH_COLOR)


        if pos_red != None and pos_blue != None and pos_yellow != None and pos_green != None:
            h, w = frame.shape[:2]
            pos_red = np.array(pos_red)
            pos_blue = np.array(pos_blue)
            pos_yellow = np.array(pos_yellow)
            pos_green = np.array(pos_green)

            dobot_pts  = np.float32([ dobot_red, dobot_yellow, dobot_blue, dobot_green ])
            camera_pts = np.float32([ pos_red, pos_yellow, pos_blue, pos_green ])

            # カメラ座標系からロボット座標系への変換行列を作成する（3-4）
            M = cv2.getPerspectiveTransform(camera_pts, dobot_pts)
            dst = cv2.warpPerspective(frame, M, (h, w))

        # 加工済の画像を表示する
        cv2.imshow('Edited Frame', edframe)

        # キー入力を1ms待つ
        k = cv2.waitKey(1)

        # 「ESC（27）」キーを押す（3-5）
        # プログラムを終了する
        if k == 27:
            break

        # 「C」キーを押す（3-6）
        # WEBカメラのゲイン値、露出の値を調整する
        elif k == ord('c'):
            g = input("gain     : ")
            e = input("exposure : ")
            print("\n - - - - - - - - - - ")
            camset.camera_set(cv2, cap, gain = float(g), exposure = float(e))
            camset.camera_get(cv2, cap)
            print(" - - - - - - - - - - \n")

        # 「S」キーを押す（3-7）
        # 変換行列を作成して、ファイルに出力する
        elif k == ord('s'):
            print("\n      \tDOBOT\t\tCAMERA")
            print("      \t(  x,    y)\t(  x,   y)")
            print("RED   \t(150,    0)\t(%3d, %3d)" % (pos_red[0], pos_red[1]))
            print("BLUE  \t(300,    0)\t(%3d, %3d)" % (pos_blue[0], pos_blue[1]))
            print("YELLOW\t(150, -100)\t(%3d, %3d)" % (pos_yellow[0], pos_yellow[1]))
            print("GREEN \t(300, -100)\t(%3d, %3d)" % (pos_green[0], pos_green[1]))
            print("\n")

            print("Transform Matrix\n", M, "\n")
            with open("TransformationMatrix.py", "w") as file:
                print("# -*- coding: utf-8 -*-", file=file)
                matrix_str = "[[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]" % (M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2], M[2][0], M[2][1], M[2][2])
                print("MATRIX = " + matrix_str, file=file)
                print(" --> saved Matrix. " + str(file) + "\n")
                break


    # キャプチャをリリースして、ウィンドウをすべて閉じる（3-8）
    cap.release()
    cv2.destroyAllWindows()
