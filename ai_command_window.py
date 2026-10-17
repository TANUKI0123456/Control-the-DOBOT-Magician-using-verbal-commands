'''
Project Name  : dobot_ai
File Name     : ai_command_window.py
File Encoding : UTF-8

生成AI（Claude）を使って、日本語の指示で DOBOT を動かすための窓。
  例：「いちごのボトルを取ってください」
      「ミルクを全部どかして」
      「赤色の部品をピックアップしてください」（--mode color のとき）

部品の見つけ方（--mode）:
  model : train.py で学習したモデルで種類を判定する（classifier.py と同じ方法）。既定値
  color : 赤・青・緑・黄の色で探す

使い方（dobot_ai フォルダで実行すること。DobotDll.dll を相対パスで読み込むため）:
  python ai_command_window.py              # DOBOT に接続して実行
  python ai_command_window.py --dry-run    # DOBOT なしで動作確認（動作はログ表示のみ）

事前準備:
  pip install anthropic pillow
  環境変数 ANTHROPIC_API_KEY に API キーを設定しておく
'''

import os
import json
import math
import queue
import threading
import time
from argparse import ArgumentParser
import tkinter as tk
from tkinter import scrolledtext

import cv2
import numpy as np
from PIL import Image, ImageTk
import anthropic

import cameraSetting as camset
from common import *
from TransformationMatrix import MATRIX
MATRIX = np.array(MATRIX)


# 使用する生成AIモデル
MODEL = "claude-opus-5-5"

# 部品の説明（生成AIに伝える。環境に合わせて書き換える）
PARTS_DESCRIPTION = "黒いマットの上に置かれた飲料の容器（アルミ缶・スチール缶・PETボトル・中身の種類ごとのボトルなど）"

# ラベルの日本語名（生成AIが日本語の指示と対応付けやすくなる。無いラベルはそのまま使う）
LABEL_JP = {
    "AL":     "アルミ缶",
    "steel":  "スチール缶",
    "pet":    "PETボトル",
    "ichigo": "いちご",
    "kinako": "きなこ",
    "milk":   "ミルク",
    "red":    "赤",
    "blue":   "青",
    "green":  "緑",
    "yellow": "黄",
}

# 色で探すときのHSV範囲（赤は色相が 0 付近と 180 付近にまたがるので 2 範囲で抽出する）
COLOR_RANGES = {
    "red":    [(RED_LOW_COLOR, RED_HIGH_COLOR), (np.array([160, 75, 75]), np.array([179, 255, 255]))],
    "blue":   [(BLUE_LOW_COLOR, BLUE_HIGH_COLOR)],
    "green":  [(GREEN_LOW_COLOR, GREEN_HIGH_COLOR)],
    "yellow": [(YELLOW_LOW_COLOR, YELLOW_HIGH_COLOR)],
}
DRAW_LABEL_COLORS = [draw_red, draw_blue, draw_green, draw_yellow, draw_white]

# 置き場所（ロボット座標系 [mm]）　環境に合わせて変更する
# ラベルの順番（0, 1, 2, ...）に SLOT_POS の位置が割り当てられる
SLOT_POS = [(200, 100), (250, 100), (150, 150), (200, 150), (250, 150), (150, 200), (200, 200)]
DEFAULT_PLACE = (300, 0)
# カメラに映らない待機位置
PARK_POS = (150, 100)
# 部品をつかむ高さ／置く高さ（オブジェクトの大きさ、環境に合わせて変更する）
PICK_Z  = -30
PLACE_Z = -30
# 部品の上で一度止まる高さ（ここからまっすぐ下ろす）
APPROACH_Z = 0
# グリッパーの向きの補正 [度]
# ボトルの胴を横からつかむように、ボトルの長い方向 + この角度 にグリッパーを回す。
# ボトルを縦方向（キャップとお尻）でつかもうとする場合は 0 に変更する。
GRIPPER_R_OFFSET = 90
# グリッパーの開閉にかかる時間 [秒]
GRIPPER_WAIT = 1.0
# DOBOT のアームが届く範囲（原点からの距離 [mm]）
REACH_MIN = 140
REACH_MAX = 320
# 「全部」と指示されたときの上限個数
MAX_PICK_ALL = 10
# 部品を探し直す間隔 [秒]（モデルの判定は重いので毎フレームは行わない）
DETECT_INTERVAL = 0.3

# カメラの明るさ設定の保存先（窓のスライダーで変えた値を次回も使う）
CAMERA_SETTINGS_FILE = "camera_settings.json"
DEFAULT_CAMERA_SETTINGS = {
    "auto_exposure": False,  # 自動露出（ON だと白飛びしやすい）
    "exposure":      -6,     # 露出（DirectShow では -13〜0。小さいほど暗い）
    "gain":          0,      # ゲイン（小さいほど暗い。自動露出のときは 192 前後になっていた）
    "soft_gain":     1.0,    # 映像の明るさ補正（1.0 = そのまま、小さいほど暗い）
}
# この明るさ以上の画素を「白飛び」として数える
CLIP_LEVEL = 250


def jp(label):
    return LABEL_JP.get(label, label)


# ---------------------------------------------------------------------------
# 画像処理
# ---------------------------------------------------------------------------
def transform_coordinate(pos_x, pos_y):
    '''
    カメラ座標系をロボット座標系に変換する
    '''
    pos = np.array([[[pos_x, pos_y]]], dtype='float32')
    transform_pos = cv2.perspectiveTransform(pos, MATRIX)
    return int(transform_pos[0][0][0]), int(transform_pos[0][0][1])


def robot_axis_angle(rect):
    '''
    回転を考慮した外接矩形（minAreaRect）の長い辺の向きを、ロボット座標系の角度 [度] で返す
    '''
    box = cv2.boxPoints(rect)
    edges = [(box[0], box[1]), (box[1], box[2])]
    p, q = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    x1, y1 = transform_coordinate(*p)
    x2, y2 = transform_coordinate(*q)
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def gripper_r(axis_angle):
    '''
    部品の長い方向の角度から、グリッパーの回転角 r を求める（-90〜90度に収める）
    グリッパーは左右対称なので 180 度回しても同じ
    '''
    r = axis_angle + GRIPPER_R_OFFSET
    while r > 90:
        r -= 180
    while r < -90:
        r += 180
    return r


def rect_preprocess(img):
    '''
    切り取った矩形の長辺に合わせて短辺を伸ばす（classifier.py と同じ）
    '''
    h, w, c = img.shape
    longest_edge = max(h, w)
    top = bottom = left = right = 0
    if h < longest_edge:
        diff_h = longest_edge - h
        top = diff_h // 2
        bottom = diff_h - top
    elif w < longest_edge:
        diff_w = longest_edge - w
        left = diff_w // 2
        right = diff_w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


class ModelDetector:
    '''
    学習済みモデルで部品を見つけて種類を判定する（classifier.py と同じ処理）
    '''
    def __init__(self, modelname, data_dir, pic_size, min_area, max_area):
        from tensorflow.keras.models import model_from_json
        self.model = model_from_json(open(MODEL_DIR + modelname + ".json").read())
        self.model.load_weights(MODEL_DIR + modelname + ".weights.h5")
        # ラベルは学習時と同じく data フォルダ内のフォルダ名の順番
        self.labels = [d for d in os.listdir(data_dir)
                       if d not in (".DS_Store", OUTFILE_NAME) and os.path.isdir(os.path.join(data_dir, d))]
        self.pic_size = pic_size
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        found = []
        images = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or self.max_area < area:
                continue
            # 回転を考慮して切り出す
            min_rect = cv2.minAreaRect(contour)
            center, size, angle = min_rect
            center = tuple(map(int, center))
            size   = tuple(map(int, size))
            if size[0] <= 0 or size[1] <= 0:
                continue
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(frame, rot_mat, (w, h))
            cropped = cv2.getRectSubPix(rotated, size, center)
            image = cv2.resize(rect_preprocess(cropped), dsize=(self.pic_size, self.pic_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32") / 255.
            images.append(image)
            found.append((center, int(area), cv2.boundingRect(contour), min_rect))

        if not images:
            return []
        # まとめて判定する
        proba = self.model.predict_on_batch(np.array(images))
        proba = np.asarray(proba)

        objects = []
        for (center, area, rect, min_rect), p in zip(found, proba):
            n = int(np.argmax(p))
            objects.append({
                "label": self.labels[n] if n < len(self.labels) else str(n),
                "score": float(p[n]),
                "cam":   center,
                "robot": transform_coordinate(*center),
                "angle": robot_axis_angle(min_rect),
                "area":  area,
                "rect":  rect,
                "box":   np.intp(cv2.boxPoints(min_rect)),
            })
        return objects


class ColorDetector:
    '''
    赤・青・緑・黄の色で部品を見つける
    '''
    def __init__(self, min_area, max_area):
        self.labels = list(COLOR_RANGES.keys())
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        objects = []
        for color, ranges in COLOR_RANGES.items():
            mask = None
            for low, high in ranges:
                m = cv2.inRange(hsv, low, high)
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area or self.max_area < area:
                    continue
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                min_rect = cv2.minAreaRect(contour)
                objects.append({
                    "label": color,
                    "score": 1.0,
                    "cam":   (cx, cy),
                    "robot": transform_coordinate(cx, cy),
                    "angle": robot_axis_angle(min_rect),
                    "area":  int(area),
                    "rect":  cv2.boundingRect(contour),
                    "box":   np.intp(cv2.boxPoints(min_rect)),
                })
        return objects


def number_objects(objects):
    '''
    大きいものから順に番号を振る
    '''
    objects.sort(key=lambda o: -o["area"])
    for i, o in enumerate(objects):
        o["id"] = i + 1
    return objects


def draw_objects(frame, objects, labels):
    '''
    検出結果をフレームに描画する
    '''
    for o in objects:
        x, y, w, h = o["rect"]
        idx = labels.index(o["label"]) if o["label"] in labels else 0
        c = DRAW_LABEL_COLORS[idx % len(DRAW_LABEL_COLORS)]
        # 回転を考慮した外接矩形（ボトルの向き）を描画
        cv2.drawContours(frame, [o["box"]], 0, c, 2)
        cv2.drawMarker(frame, o["cam"], c, cv2.MARKER_TILTED_CROSS, thickness=2)
        label = "#%d %s %.0f%% (%d, %d)" % (o["id"], o["label"], o["score"] * 100, o["robot"][0], o["robot"][1])
        cv2.rectangle(frame, (x, max(y - 16, 0)), (x + len(label) * 8, y), draw_white, -1)
        cv2.putText(frame, label, (x, max(y - 4, 12)), font, FONT_SIZE, draw_black, FONT_WIDTH, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 生成AI（日本語の指示 → 動作リスト）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """あなたはロボットアーム DOBOT Magician の操作アシスタントです。
ユーザーの日本語の指示を、ロボットが実行できる動作のリストに変換してください。
（先端の道具: {tool}。「つかんで」「吸って」などの言い方の違いは気にせず、運ぶ指示は pick にする）

扱う部品: {parts}
部品の種類（ラベル）: {labels}

使える動作 (action):
- pick : 指定した種類の部品を吸着して持ち上げ、置き場所へ運ぶ。
         target = 部品の種類（上のラベル名）。種類を限定しない言い方（「部品」「どれか」「何でも」など）のときは any
         count  = 運ぶ個数（「全部」「すべて」のときは 0、指定がなければ 1）
         place  = 置き場所（指定なしは auto＝部品の種類ごとの置き場。「いちごの置き場へ」ならそのラベル名。
                  「端に」「よけておいて」など種類と関係ない場所なら default）
- home : ホームポジションへ戻る（原点合わせ）。target は any、count は 0、place は auto にする。

ルール:
- 複数の指示は順番どおりに actions に並べる。
- 現在カメラに映っている部品の一覧が与えられる。映っていない種類の部品を指示された場合は actions を空にして、reply でそのことを伝える。
- 何をすればよいか分からない指示、ロボットにできない指示、雑談や質問には actions を空にして reply で答える。推測で動かさない。
- reply にはユーザーへの短い日本語の返答（これから何をするか、または動かない理由）を書く。
"""


def build_schema(labels):
    return {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["pick", "home"]},
                        "target": {"type": "string", "enum": labels + ["any"]},
                        "count":  {"type": "integer"},
                        "place":  {"type": "string", "enum": labels + ["auto", "default"]},
                    },
                    "required": ["action", "target", "count", "place"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["reply", "actions"],
        "additionalProperties": False,
    }


def describe_scene(objects):
    '''
    検出した部品の一覧を生成AIに渡す文章にする
    '''
    if not objects:
        return "現在カメラに映っている部品: なし"
    lines = ["現在カメラに映っている部品:"]
    for o in objects:
        lines.append("- #%d %s（%s） 確信度 %.0f%% ロボット座標 x=%d, y=%d" %
                     (o["id"], o["label"], jp(o["label"]), o["score"] * 100, o["robot"][0], o["robot"][1]))
    return "\n".join(lines)


def interpret_command(client, text, objects, labels, tool):
    '''
    日本語の指示を生成AIで解釈し、{"reply": ..., "actions": [...]} を返す
    '''
    system = SYSTEM_PROMPT.format(
        tool="グリッパー（開閉してつかむ）" if tool == "gripper" else "吸引カップ",
        parts=PARTS_DESCRIPTION,
        labels="、".join("%s（%s）" % (l, jp(l)) for l in labels),
    )
    user_content = describe_scene(objects) + "\n\nユーザーの指示: " + text
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        # 安全判定で断られたときに自動で別モデルへ切り替える
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=system,
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": build_schema(labels)},
        },
        messages=[{"role": "user", "content": user_content}],
    )
    if response.stop_reason == "refusal":
        return {"reply": "この指示は実行できません。", "actions": []}
    if response.stop_reason == "max_tokens":
        return {"reply": "AIの応答が途中で切れました。もう一度試してください。", "actions": []}
    text_out = next((b.text for b in response.content if b.type == "text"), None)
    if text_out is None:
        return {"reply": "AIから応答がありませんでした。", "actions": []}
    return json.loads(text_out)


# ---------------------------------------------------------------------------
# ロボット操作
# ---------------------------------------------------------------------------
class Robot:
    '''
    DOBOT の操作。dry_run のときは実機を動かさず、ログだけ出す
    '''
    def __init__(self, dry_run, log, tool="gripper"):
        self.dry_run = dry_run
        self.log = log
        self.tool = tool
        if not dry_run:
            import dobotClassifier as dc
            import myDobotModule as dobot
            self.dc = dc
            self.dobot = dobot

    def initialize(self):
        if self.dry_run:
            self.log("[ドライラン] DOBOT には接続しません")
        else:
            self.dc.initialize()

    def finalize(self):
        if not self.dry_run:
            self.dc.finalize()

    def move_home(self):
        self.log("ホームポジションへ移動します")
        if not self.dry_run:
            self.dc.move_home()

    def pick_and_place(self, pos, place, axis_angle=0.0):
        if self.tool == "gripper":
            self.gripper_pick_and_place(pos, place, axis_angle)
        else:
            self.suction_pick_and_place(pos, place)

    def gripper_pick_and_place(self, pos, place, axis_angle):
        '''
        グリッパーを開いて部品の上から下ろし、閉じてつかんで運ぶ
        '''
        x, y = pos
        px, py = place
        r = gripper_r(axis_angle)
        self.log("  (%d, %d) の部品をグリッパーでつかんで (%d, %d) へ運びます（向き r=%.0f度）" %
                 (x, y, px, py, r))
        if self.dry_run:
            time.sleep(1.0)
            return
        d = self.dobot
        # 部品の真上でグリッパーを部品の向きに合わせ、開いておく
        d.move(x, y, APPROACH_Z, r)
        d.gripper(True, False)
        time.sleep(GRIPPER_WAIT)
        # 下降して閉じる（つかむ）→ 上昇
        d.move(x, y, PICK_Z, r)
        d.gripper(True, True)
        time.sleep(GRIPPER_WAIT)
        d.move(x, y, APPROACH_Z, r)
        # 置き場の真上 → 下降して開く（離す）→ 上昇
        d.move(px, py, APPROACH_Z, r)
        d.move(px, py, PLACE_Z, r)
        d.gripper(True, False)
        time.sleep(GRIPPER_WAIT)
        d.move(px, py, APPROACH_Z, r)
        # エアーポンプを止めて、カメラに映らない場所へ
        d.gripper(False, False)
        d.move(PARK_POS[0], PARK_POS[1], 0, 0)

    def suction_pick_and_place(self, pos, place):
        '''
        吸引カップで部品を吸着して運ぶ
        '''
        x, y = pos
        px, py = place
        self.log("  (%d, %d) の部品を吸着して (%d, %d) へ運びます" % (x, y, px, py))
        if self.dry_run:
            time.sleep(1.0)
            return
        d = self.dobot
        # 部品の真上 → 下降して吸着 → 上昇
        d.move(x, y, APPROACH_Z, 0)
        d.move(x, y, PICK_Z, 0)
        d.suctioncup(True, True)
        time.sleep(0.5)
        d.move(x, y, APPROACH_Z, 0)
        # 置き場の真上 → 下降して離す → 上昇
        d.move(px, py, APPROACH_Z, 0)
        d.move(px, py, PLACE_Z, 0)
        d.suctioncup(False, False)
        time.sleep(0.5)
        d.move(px, py, APPROACH_Z, 0)
        # カメラに映らない場所へ
        d.move(PARK_POS[0], PARK_POS[1], 0, 0)


def reachable(pos):
    r = math.hypot(pos[0], pos[1])
    return REACH_MIN <= r <= REACH_MAX


# ---------------------------------------------------------------------------
# カメラの明るさ
# ---------------------------------------------------------------------------
def open_camera(index):
    '''
    カメラを開く。Windows では露出を変えられる DirectShow を優先して使う
    '''
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap, "DirectShow"
    cap.release()
    return cv2.VideoCapture(index), "標準"


def load_camera_settings():
    settings = dict(DEFAULT_CAMERA_SETTINGS)
    try:
        with open(CAMERA_SETTINGS_FILE, encoding="utf-8") as f:
            settings.update(json.load(f))
    except (OSError, ValueError):
        pass
    return settings


def save_camera_settings(settings):
    try:
        with open(CAMERA_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 窓（GUI）
# ---------------------------------------------------------------------------
class App:
    def __init__(self, root, args, detector):
        self.root = root
        self.args = args
        self.detector = detector
        self.labels = detector.labels
        self.ui_queue = queue.Queue()
        self.lock = threading.Lock()
        self.latest_objects = []
        self.last_detect = 0.0
        self.busy = False
        self.cancel = False

        # ラベルごとの置き場所
        self.place_pos = {"default": DEFAULT_PLACE}
        for i, l in enumerate(self.labels):
            self.place_pos[l] = SLOT_POS[i] if i < len(SLOT_POS) else DEFAULT_PLACE

        root.title("DOBOT AI 指示ウィンドウ")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 左：カメラ映像
        left = tk.Frame(root)
        left.pack(side=tk.LEFT, padx=8, pady=8)
        self.video = tk.Label(left, bg="black")
        self.video.pack()
        self.scene_label = tk.Label(left, text="", justify=tk.LEFT, anchor="w", font=("Meiryo", 10))
        self.scene_label.pack(fill=tk.X, pady=(6, 0))
        self.build_brightness_panel(left)

        # 右：指示入力とログ
        right = tk.Frame(root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        tk.Label(right, text="日本語で指示を入力してください（Enter で実行）",
                 font=("Meiryo", 11, "bold")).pack(anchor="w")
        self.entry = tk.Entry(right, font=("Meiryo", 13), width=40)
        self.entry.pack(fill=tk.X, pady=4)
        self.entry.bind("<Return>", lambda e: self.on_submit())
        self.entry.focus_set()

        buttons = tk.Frame(right)
        buttons.pack(fill=tk.X)
        self.run_button = tk.Button(buttons, text="実行", width=10, command=self.on_submit)
        self.run_button.pack(side=tk.LEFT)
        tk.Button(buttons, text="停止（次の部品から）", command=self.on_stop).pack(side=tk.LEFT, padx=4)

        tk.Label(right, text="例文（クリックで入力）:", font=("Meiryo", 9)).pack(anchor="w", pady=(8, 0))
        first = jp(self.labels[0]) if self.labels else "部品"
        examples = [
            "%sを取ってください" % first,
            "%sを全部どかして" % first,
            "どんな部品が見えますか？",
            "ホームに戻って",
        ]
        for ex in examples:
            tk.Button(right, text=ex, anchor="w", relief=tk.FLAT, fg="#1a5fb4",
                      command=lambda t=ex: self.set_text(t)).pack(fill=tk.X)

        self.log_box = scrolledtext.ScrolledText(right, height=18, font=("Meiryo", 10), state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.log("部品の種類: " + "、".join("%s（%s）" % (l, jp(l)) for l in self.labels))

        # 生成AIのクライアント
        try:
            self.client = anthropic.Anthropic()
        except Exception as e:
            self.client = None
            self.log("生成AIの初期化に失敗しました: %s" % e)
            self.log("環境変数 ANTHROPIC_API_KEY を設定してください。")

        # カメラ
        self.cap, backend = open_camera(args.camera)
        if not self.cap.isOpened():
            self.log("カメラ %d を開けませんでした。--camera で番号を指定してください。" % args.camera)
        else:
            self.log("カメラ %d を開きました（%s）" % (args.camera, backend))
        self.apply_camera_settings()
        camset.camera_get(cv2, self.cap)

        # DOBOT
        self.robot = Robot(args.dry_run, self.log_from_worker, args.tool)
        self.log("先端の道具: " + ("グリッパー" if args.tool == "gripper" else "吸引カップ"))
        self.robot.initialize()

        self.log("準備ができました。指示を入力してください。")
        self.update_frame()
        self.poll_queue()

    # --- ログ -----------------------------------------------------------
    def log(self, text):
        self.log_box.configure(state=tk.NORMAL)
        self.log_box.insert(tk.END, text + "\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state=tk.DISABLED)

    def log_from_worker(self, text):
        # 別スレッドから呼ばれても安全なように、キュー経由で表示する
        self.ui_queue.put(("log", text))

    def poll_queue(self):
        try:
            while True:
                kind, value = self.ui_queue.get_nowait()
                if kind == "log":
                    self.log(value)
                elif kind == "done":
                    self.busy = False
                    self.run_button.configure(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.root.after(50, self.poll_queue)

    # --- 明るさ調整 -----------------------------------------------------
    def build_brightness_panel(self, parent):
        '''
        カメラ映像の下に、明るさ調整のスライダーを並べる
        '''
        self.cam_settings = load_camera_settings()
        s = self.cam_settings

        panel = tk.LabelFrame(parent, text="カメラの明るさ（白飛びするときは左へ動かす）", font=("Meiryo", 10))
        panel.pack(fill=tk.X, pady=(6, 0))

        self.clip_label = tk.Label(panel, text="", anchor="w", font=("Meiryo", 10))
        self.clip_label.grid(row=0, column=0, columnspan=3, sticky="w")

        self.auto_var = tk.BooleanVar(value=s["auto_exposure"])
        tk.Checkbutton(panel, text="自動露出（外すと「露出」で調整できる）", variable=self.auto_var,
                       command=self.on_brightness_change).grid(row=1, column=0, columnspan=2, sticky="w")
        tk.Button(panel, text="詳細設定…", command=self.open_driver_settings).grid(row=1, column=2, sticky="e")

        def add_scale(row, text, frm, to, res, value):
            tk.Label(panel, text=text, font=("Meiryo", 9)).grid(row=row, column=0, sticky="w")
            scale = tk.Scale(panel, from_=frm, to=to, resolution=res, orient=tk.HORIZONTAL, length=380)
            scale.set(value)
            scale.configure(command=lambda v: self.on_brightness_change())
            scale.grid(row=row, column=1, columnspan=2, sticky="we")
            return scale

        self.exposure_scale  = add_scale(2, "露出（カメラ）",   -13, 0,   1,    s["exposure"])
        self.gain_scale      = add_scale(3, "ゲイン（カメラ）",   0, 255, 1,    s["gain"])
        self.soft_gain_scale = add_scale(4, "明るさ補正（映像）", 0.2, 1.5, 0.05, s["soft_gain"])
        tk.Button(panel, text="初期値に戻す", command=self.reset_brightness).grid(row=5, column=2, sticky="e")
        self.update_exposure_state()

    def update_exposure_state(self):
        self.exposure_scale.configure(state=tk.DISABLED if self.auto_var.get() else tk.NORMAL)

    def on_brightness_change(self):
        s = self.cam_settings
        s["auto_exposure"] = bool(self.auto_var.get())
        s["exposure"]      = int(self.exposure_scale.get())
        s["gain"]          = int(self.gain_scale.get())
        s["soft_gain"]     = float(self.soft_gain_scale.get())
        self.update_exposure_state()
        self.apply_camera_settings()
        save_camera_settings(s)

    def reset_brightness(self):
        self.auto_var.set(DEFAULT_CAMERA_SETTINGS["auto_exposure"])
        self.exposure_scale.set(DEFAULT_CAMERA_SETTINGS["exposure"])
        self.gain_scale.set(DEFAULT_CAMERA_SETTINGS["gain"])
        self.soft_gain_scale.set(DEFAULT_CAMERA_SETTINGS["soft_gain"])
        self.on_brightness_change()

    def apply_camera_settings(self):
        '''
        露出・ゲインをカメラに設定する（カメラによっては効かないものもある）
        '''
        if not hasattr(self, "cap"):
            return
        s = self.cam_settings
        if s["auto_exposure"]:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # DirectShow: 0.75 = 自動
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # DirectShow: 0.25 = 手動
            self.cap.set(cv2.CAP_PROP_EXPOSURE, s["exposure"])
        self.cap.set(cv2.CAP_PROP_GAIN, s["gain"])

    def open_driver_settings(self):
        '''
        カメラのドライバーの設定画面を開く（DirectShow のときに使える）
        '''
        if not self.cap.set(cv2.CAP_PROP_SETTINGS, 1):
            self.log("このカメラでは詳細設定の画面を開けませんでした。")

    def adjust_brightness(self, frame):
        '''
        映像の明るさ補正（カメラ側で暗くできないときの予備）
        '''
        g = self.cam_settings["soft_gain"]
        if abs(g - 1.0) < 1e-3:
            return frame
        return cv2.convertScaleAbs(frame, alpha=g, beta=0)

    # --- カメラ映像 -----------------------------------------------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.adjust_brightness(frame)
            # 白飛びしている画素の割合を表示する
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clip = float(np.count_nonzero(gray >= CLIP_LEVEL)) / gray.size * 100
            self.clip_label.configure(
                text="白飛び: %.1f%%  %s" % (clip, "← 暗くしてください" if clip >= 1.0 else "OK"),
                fg="#c01c28" if clip >= 1.0 else "#26a269")

            now = time.time()
            if now - self.last_detect >= DETECT_INTERVAL:
                self.last_detect = now
                objects = number_objects(self.detector.detect(frame))
                with self.lock:
                    self.latest_objects = objects
                counts = {}
                for o in objects:
                    counts[o["label"]] = counts.get(o["label"], 0) + 1
                summary = "、".join("%s %d個" % (jp(l), n) for l, n in counts.items()) or "なし"
                self.scene_label.configure(text="検出中の部品: " + summary)

            draw_objects(frame, self.get_objects(), self.labels)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video.configure(image=img)
            self.video.image = img
        self.root.after(30, self.update_frame)

    def get_objects(self):
        with self.lock:
            return list(self.latest_objects)

    # --- 操作 -----------------------------------------------------------
    def set_text(self, text):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)
        self.entry.focus_set()

    def on_stop(self):
        if self.busy:
            self.cancel = True
            self.log("停止します（いま運んでいる部品を置いたら止まります）")

    def on_submit(self):
        text = self.entry.get().strip()
        if not text or self.busy:
            return
        if self.client is None:
            self.log("生成AIが使えません。ANTHROPIC_API_KEY を確認してください。")
            return
        self.entry.delete(0, tk.END)
        self.log("\nあなた: " + text)
        self.busy = True
        self.cancel = False
        self.run_button.configure(state=tk.DISABLED)
        threading.Thread(target=self.worker, args=(text,), daemon=True).start()

    def worker(self, text):
        log = self.log_from_worker
        try:
            log("（AIが指示を解釈しています…）")
            plan = interpret_command(self.client, text, self.get_objects(), self.labels, self.args.tool)
            log("AI: " + plan["reply"])
            for act in plan["actions"]:
                if self.cancel:
                    break
                if act["action"] == "home":
                    self.robot.move_home()
                elif act["action"] == "pick":
                    self.do_pick(act["target"], act["count"], act["place"])
            log("停止しました。" if self.cancel else "完了しました。")
        except anthropic.AuthenticationError:
            log("APIキーが正しくありません。ANTHROPIC_API_KEY を確認してください。")
        except anthropic.APIConnectionError:
            log("生成AIに接続できません。ネットワークを確認してください。")
        except anthropic.APIStatusError as e:
            if "credit balance" in str(e.message):
                log("APIのクレジット残高が足りません。Claude Console の Billing でクレジットを購入してください。")
            else:
                log("生成AIのエラー (%d): %s" % (e.status_code, e.message))
        except Exception as e:
            log("エラー: %s" % e)
        finally:
            self.ui_queue.put(("done", None))

    def do_pick(self, target, count, place):
        log = self.log_from_worker
        goal = MAX_PICK_ALL if count <= 0 else count
        name = "部品" if target == "any" else jp(target)
        done = 0
        while done < goal and not self.cancel:
            candidates = [o for o in self.get_objects()
                          if (target == "any" or o["label"] == target) and reachable(o["robot"])]
            if not candidates:
                if done == 0:
                    log("  %s（アームが届く範囲）が見つかりません。" % name)
                break
            obj = candidates[0]
            dest = self.place_pos[obj["label"]] if place == "auto" else self.place_pos[place]
            log("%s #%d をピックアップします (%d/%s)" %
                (jp(obj["label"]), obj["id"], done + 1, "全部" if count <= 0 else str(count)))
            self.robot.pick_and_place(obj["robot"], dest, obj["angle"])
            done += 1
            # アームが画面外へ出たあとの映像で、次の部品を探し直す
            time.sleep(1.5)
        if done:
            log("  %d個運びました。" % done)

    def on_close(self):
        self.cancel = True
        try:
            self.robot.finalize()
        finally:
            self.cap.release()
            self.root.destroy()


def get_option():
    argparser = ArgumentParser()
    argparser.add_argument("--mode", dest="mode", choices=["model", "color"], default="model",
                           help="model: trained model / color: red, blue, green, yellow.")
    argparser.add_argument("--tool", dest="tool", choices=["gripper", "suction"], default="gripper",
                           help="gripper: open/close gripper / suction: suction cup.")
    argparser.add_argument("--camera", dest="camera", type=int, default=1,
                           help="Camera index.")
    argparser.add_argument("--dry-run", dest="dry_run", action="store_true",
                           help="Do not connect to DOBOT (log only).")
    argparser.add_argument("-d", "--directory", dest="load_dir", type=str, default=DATA_DIR,
                           help="Directory that contains data used for learning.")
    argparser.add_argument("-m", "--modelname", dest="modelname", type=str, default="model",
                           help="Input model name. (***.json, ***.weights.h5)")
    argparser.add_argument("-s", "--size", dest="pic_size", type=int, default=PIC_SIZE,
                           help="Picture size when training.")
    argparser.add_argument("--min", dest="min_area_size", type=int, default=MIN_AREA_SIZE,
                           help="Minimum area size.")
    argparser.add_argument("--max", dest="max_area_size", type=int, default=MAX_AREA_SIZE,
                           help="Maximum area size.")
    return argparser.parse_args()


if __name__ == '__main__':
    args = get_option()
    if args.mode == "model":
        print("\n - - - model data loading ( " + args.modelname + ".json ) - - - \n")
        detector = ModelDetector(args.modelname, args.load_dir, args.pic_size,
                                 args.min_area_size, args.max_area_size)
    else:
        detector = ColorDetector(args.min_area_size, args.max_area_size)
    root = tk.Tk()
    App(root, args, detector)
    root.mainloop()
