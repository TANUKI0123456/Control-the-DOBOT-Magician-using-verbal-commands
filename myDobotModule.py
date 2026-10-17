'''
Project Name  : dobot_ai
File Name     : myDobotModule.py
File Encoding : UTF-8
Copyright © 2020 Afrel Co.,Ltd.
'''

import DobotDllType as dType
from DobotDllType import PTPMode, JC, DobotConnect

api = None

def initialize():
    '''
    DOBOTの初期化処理
    '''
    # APIをロードし、DOBOTに接続する
    global api
    api = dType.load()

    state = dType.ConnectDobot(api, "COM7", 115200)[0]
    if not state == DobotConnect.DobotConnect_NoError:
        print("Could not connect to DOBOT")
        exit()

    # 初期設定
    dType.SetCmdTimeout(api, 3000)
    dType.SetQueuedCmdClear(api)
    dType.SetQueuedCmdStartExec(api)
    deviceName = "DOBOT Magician"
    dType.SetDeviceName(api, deviceName)

    dType.SetJOGJointParams(api, 50, 50, 50, 50, 50, 50, 50, 50, True)
    dType.SetJOGCoordinateParams(api, 50, 50, 50, 50, 50, 50, 50, 50, True)
    dType.SetJOGCommonParams(api, 100, 100, True)
    dType.SetPTPJointParams(api, 100, 100, 100, 100, 100, 100, 100, 100, True)
    dType.SetPTPCoordinateParams(api, 100, 100, 100, 100, True)
    dType.SetPTPJumpParams(api, 20, 100, True)
    dType.SetPTPCommonParams(api, 30, 30, True)
    dType.SetHOMEParams(api, 200, 0, 0, 0, True)
    dType.SetEndEffectorParams(api, 59.7 ,0, 0, 0)

def move_home():
    '''
    アームをホームポジションへ移動
    '''
    dType.SetHOMECmdEx(api, 0, True)

def finalize():
    '''
    DOBOTの終了処理
    '''
    gripper(False, False)
    suctioncup(False, False)
    dType.DisconnectDobot(api)

def move(x, y, z, r):
    '''
    DOBOTを指定座標に動かす。
    モードは「PTPMOVJXYZMode」で固定。
    '''
    dType.SetPTPCmdEx(api, PTPMode.PTPMOVJXYZMode, x, y, z, r, True)

def gripper(on, grip):
    '''
    グリッパーとエアーポンプのON/OFFを制御する。
    on   bool: True→電源ON、False→電源OFF
    grip bool: True→閉じる、False→開く
    '''
    dType.SetEndEffectorGripperEx(api, on, grip)

def suctioncup(on, suck):
    '''
    吸引カップとエアーポンプのON/OFFを制御する。
    on   bool: True→電源ON、False→電源OFF
    grip bool: True→吸う、False→吐く
    '''
    dType.SetEndEffectorSuctionCupEx(api, on, suck)

def wait(sec):
    '''
    指定した秒数だけ待機する。
    sec int: 待機する秒数
    '''
    dType.SetWAITCmdEx(api, sec, 1)
