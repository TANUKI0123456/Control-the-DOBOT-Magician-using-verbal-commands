'''
Project Name  : dobot_ai
File Name     : cameraSetting.py
File Encoding : UTF-8
Copyright © 2020 Afrel Co.,Ltd.
'''

#set the width and height, and UNSUCCESSFULLY set the exposure time
#  0. CV_CAP_PROP_POS_MSEC        ビデオファイルの現在位置（ミリ秒）。
#  1. CV_CAP_PROP_POS_FRAMES      次にデコード/キャプチャされるフレームの0から始まるインデックス。
#  2. CV_CAP_PROP_POS_AVI_RATIO   ビデオファイルの相対位置
#  3. CV_CAP_PROP_FRAME_WIDTH     ビデオストリーム内のフレームの幅。
#  4. CV_CAP_PROP_FRAME_HEIGHT    ビデオストリーム内のフレームの高さ。
#  5. CV_CAP_PROP_FPS             フレームレート。
#  6. CV_CAP_PROP_FOURCC          コーデックの4文字のコード。
#  7. CV_CAP_PROP_FRAME_COUNT     ビデオファイルのフレーム数。
#  8. CV_CAP_PROP_FORMAT          retrieve（）によって返されるMatオブジェクトのフォーマット。
#  9. CV_CAP_PROP_MODE            現在のキャプチャモードを示すバックエンド固有の値。
# 10. CV_CAP_PROP_BRIGHTNESS      画像の明るさ（カメラのみ）。
# 11. CV_CAP_PROP_CONTRAST        画像のコントラスト（カメラのみ）。
# 12. CV_CAP_PROP_SATURATION      画像の彩度（カメラのみ）。
# 13. CV_CAP_PROP_HUE             画像の色相（カメラのみ）。
# 14. CV_CAP_PROP_GAIN            画像のゲイン（カメラのみ）。
# 15. CV_CAP_PROP_EXPOSURE        露出（カメラのみ）
# 16. CV_CAP_PROP_CONVERT_RGB     画像をRGBに変換するかどうかを示すブールフラグ。
# 17. CV_CAP_PROP_WHITE_BALANCE   現在サポートされていません
# 18. CV_CAP_PROP_RECTIFICATION   ステレオカメラ用の整流フラグ（注：現在DC1394 v 2.xバックエンドでのみサポートされています）

def camera_set(cv2, cap, brightness=None, contrast=None, saturation=None, hue=None, gain=None, exposure=None):
  '''
  WEBカメラの設定をする
  '''
  # cap.set(cv2.CAP_PROP_POS_MSEC,      0)
  # cap.set(cv2.CAP_PROP_POS_FRAMES,    0)
  # cap.set(cv2.CAP_PROP_POS_AVI_RATIO, -1)
  # cap.set(cv2.CAP_PROP_FRAME_WIDTH,   640)
  # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  480)
  # cap.set(cv2.CAP_PROP_FPS,           30)
  # cap.set(cv2.CAP_PROP_FOURCC,        20)
  # cap.set(cv2.CAP_PROP_FRAME_COUNT,   -1)
  # cap.set(cv2.CAP_PROP_FORMAT,        -1)
  # cap.set(cv2.CAP_PROP_MODE,          0)
  # cap.set(cv2.CAP_PROP_BRIGHTNESS,    brightness)
  # cap.set(cv2.CAP_PROP_CONTRAST,      contrast)
  # cap.set(cv2.CAP_PROP_SATURATION,    saturation)
  # cap.set(cv2.CAP_PROP_HUE,           hue)
  cap.set(cv2.CAP_PROP_GAIN,          gain)
  cap.set(cv2.CAP_PROP_EXPOSURE,      exposure)
  # cap.set(cv2.CAP_PROP_CONVERT_RGB,   1)
  # cap.set(cv2.CAP_PROP_WHITE_BALANCE, )
  # cap.set(cv2.CAP_PROP_RECTIFICATION, -1)


def camera_get(cv2, cap):
  '''
  WEBカメラの設定情報を取得する
  '''
  # print("POS_MSEC      : " + str(cap.get(cv2.CAP_PROP_POS_MSEC)))
  # print("POS_FRAMES    : " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
  # print("POS_AVI_RATIO : " + str(cap.get(cv2.CAP_PROP_POS_AVI_RATIO)))
  print("FRAME_WIDTH   : " + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
  print("FRAME_HEIGHT  : " + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  print("FPS           : " + str(cap.get(cv2.CAP_PROP_FPS)))
  # print("FOURCC        : " + str(cap.get(cv2.CAP_PROP_FOURCC)))
  # print("FRAME_COUNT   : " + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
  # print("FORMAT        : " + str(cap.get(cv2.CAP_PROP_FORMAT)))
  # print("MODE          : " + str(cap.get(cv2.CAP_PROP_MODE)))
  print("BRIGHTNESS    : " + str(cap.get(cv2.CAP_PROP_BRIGHTNESS)))
  print("CONTRAST      : " + str(cap.get(cv2.CAP_PROP_CONTRAST)))
  print("SATURATION    : " + str(cap.get(cv2.CAP_PROP_SATURATION)))
  print("HUE           : " + str(cap.get(cv2.CAP_PROP_HUE)))
  print("GAIN          : " + str(cap.get(cv2.CAP_PROP_GAIN)))
  print("EXPOSURE      : " + str(cap.get(cv2.CAP_PROP_EXPOSURE)))
  # print("CONVERT_RGB   : " + str(cap.get(cv2.CAP_PROP_CONVERT_RGB)))
  # print("WHITE_BALANCE : " + str(cap.get(cv2.CAP_PROP_WHITE_BALANCE)))
  # print("RECTIFICATION : " + str(cap.get(cv2.CAP_PROP_RECTIFICATION)))
