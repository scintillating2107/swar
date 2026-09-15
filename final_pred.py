"""
SignBridge — ASL Fingerspelling to Text and Speech

Author: Srishti Singh
"""

import math
import traceback
from pathlib import Path

import cv2
import numpy as np

import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
from PIL import Image, ImageTk

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cnn8grps_rad1_model.h5"
STABLE_FRAMES = 4
offset = 29

try:
    import enchant
    ddd = enchant.Dict("en-US")
    ENCHANT_AVAILABLE = True
except Exception:
    ddd = None
    ENCHANT_AVAILABLE = False

hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)


def parse_hand(hand_entry):
    """Support both cvzone API versions."""
    if hand_entry is None:
        return None
    if isinstance(hand_entry, dict) and "bbox" in hand_entry:
        return hand_entry
    if isinstance(hand_entry, (list, tuple)):
        for item in hand_entry:
            parsed = parse_hand(item)
            if parsed:
                return parsed
    return None


def make_white_canvas():
    return np.ones((400, 400, 3), dtype=np.uint8) * 255


# Application :

class Application:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SignBridge — ASL to Text & Speech | Srishti Singh")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("1400x820")
        self.root.configure(bg="#1a1a2e")

        self.loading_label = tk.Label(
            self.root,
            text="Loading model, please wait...",
            font=("Segoe UI", 16),
            fg="white",
            bg="#1a1a2e",
        )
        self.loading_label.place(x=500, y=380)
        self.root.update()

        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        if not MODEL_PATH.exists():
            self.loading_label.config(text=f"Error: model not found at {MODEL_PATH.name}")
            return
        self.model = load_model(str(MODEL_PATH))
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)
        voices = self.speak_engine.getProperty("voices")
        if voices:
            self.speak_engine.setProperty("voice", voices[0].id)

        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10
        self._debounce_char = None
        self._debounce_count = 0
        self.hand_detected = False

        self.loading_label.destroy()

        self.panel = tk.Label(self.root, bg="#16213e")
        self.panel.place(x=30, y=90, width=520, height=390)

        self.panel2 = tk.Label(self.root, bg="#16213e")
        self.panel2.place(x=580, y=90, width=400, height=400)

        self.T = tk.Label(self.root, bg="#1a1a2e", fg="white")
        self.T.place(x=30, y=10)
        self.T.config(text="SignBridge — ASL to Text & Speech", font=("Segoe UI", 22, "bold"))

        self.subtitle = tk.Label(self.root, bg="#1a1a2e", fg="#a0a0b0")
        self.subtitle.place(x=30, y=48)
        self.subtitle.config(
            text="By Srishti Singh  •  Real-time recognition with spell-check and text-to-speech",
            font=("Segoe UI", 11),
        )

        self.status_label = tk.Label(self.root, bg="#1a1a2e", fg="#ff6b6b", font=("Segoe UI", 12, "bold"))
        self.status_label.place(x=30, y=500)
        self.status_label.config(text="● No hand detected")

        self.help_label = tk.Label(
            self.root,
            bg="#1a1a2e",
            fg="#7ec8e3",
            font=("Segoe UI", 10),
            justify=tk.LEFT,
            text=(
                "Gestures: Hold a letter steady → swipe right to confirm word  |  "
                "Open palm = space  |  Fist forward = backspace"
            ),
        )
        self.help_label.place(x=30, y=525)

        self.T1 = tk.Label(self.root, bg="#1a1a2e", fg="white")
        self.T1.place(x=580, y=500)
        self.T1.config(text="Character:", font=("Segoe UI", 14, "bold"))

        self.panel3 = tk.Label(self.root, bg="#1a1a2e", fg="#4ecca3")
        self.panel3.place(x=700, y=495)
        self.panel3.config(text="-", font=("Segoe UI", 36, "bold"))

        self.T3 = tk.Label(self.root, bg="#1a1a2e", fg="white")
        self.T3.place(x=30, y=560)
        self.T3.config(text="Sentence:", font=("Segoe UI", 14, "bold"))

        self.panel5 = tk.Label(self.root, bg="#1a1a2e", fg="white", anchor="w")
        self.panel5.place(x=150, y=555, width=900)

        self.T4 = tk.Label(self.root, bg="#1a1a2e", fg="#ff6b6b")
        self.T4.place(x=30, y=610)
        self.T4.config(text="Suggestions:", font=("Segoe UI", 14, "bold"))

        self.b1 = tk.Button(self.root, bg="#0f3460", fg="white", relief=tk.FLAT)
        self.b1.place(x=180, y=605, width=180, height=36)

        self.b2 = tk.Button(self.root, bg="#0f3460", fg="white", relief=tk.FLAT)
        self.b2.place(x=380, y=605, width=180, height=36)

        self.b3 = tk.Button(self.root, bg="#0f3460", fg="white", relief=tk.FLAT)
        self.b3.place(x=580, y=605, width=180, height=36)

        self.b4 = tk.Button(self.root, bg="#0f3460", fg="white", relief=tk.FLAT)
        self.b4.place(x=780, y=605, width=180, height=36)

        self.clear = tk.Button(self.root, bg="#e94560", fg="white", relief=tk.FLAT)
        self.clear.place(x=1100, y=555, width=120, height=40)
        self.clear.config(text="Clear", font=("Segoe UI", 14, "bold"), command=self.clear_fun)

        self.speak = tk.Button(self.root, bg="#4ecca3", fg="#1a1a2e", relief=tk.FLAT)
        self.speak.place(x=1240, y=555, width=120, height=40)
        self.speak.config(text="Speak", font=("Segoe UI", 14, "bold"), command=self.speak_fun)

        self.str = ""
        self.word = ""
        self.current_symbol = "-"
        self.word1 = ""
        self.word2 = ""
        self.word3 = ""
        self.word4 = ""

        self.video_loop()

    def draw_skeleton(self, white, pts, w, h):
        os_x = ((400 - w) // 2) - 15
        os_y = ((400 - h) // 2) - 15
        for t in range(0, 4):
            cv2.line(
                white,
                (pts[t][0] + os_x, pts[t][1] + os_y),
                (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y),
                (0, 255, 0),
                3,
            )
        for start in (5, 9, 13, 17):
            for t in range(start, start + 3):
                cv2.line(
                    white,
                    (pts[t][0] + os_x, pts[t][1] + os_y),
                    (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y),
                    (0, 255, 0),
                    3,
                )
        cv2.line(white, (pts[5][0] + os_x, pts[5][1] + os_y), (pts[9][0] + os_x, pts[9][1] + os_y), (0, 255, 0), 3)
        cv2.line(white, (pts[9][0] + os_x, pts[9][1] + os_y), (pts[13][0] + os_x, pts[13][1] + os_y), (0, 255, 0), 3)
        cv2.line(white, (pts[13][0] + os_x, pts[13][1] + os_y), (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
        cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), (pts[5][0] + os_x, pts[5][1] + os_y), (0, 255, 0), 3)
        cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
        for i in range(21):
            cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)
        return white

    def update_ui(self):
        display_char = self.current_symbol
        if self._debounce_count < STABLE_FRAMES and isinstance(self.current_symbol, str) and len(self.current_symbol) == 1:
            display_char = f"{self.current_symbol}~"
        self.panel3.config(text=display_char, font=("Segoe UI", 36, "bold"))
        self.panel5.config(text=self.str or " ", font=("Segoe UI", 18), wraplength=900)
        self.b1.config(text=self.word1 or " ", font=("Segoe UI", 11), command=self.action1)
        self.b2.config(text=self.word2 or " ", font=("Segoe UI", 11), command=self.action2)
        self.b3.config(text=self.word3 or " ", font=("Segoe UI", 11), command=self.action3)
        self.b4.config(text=self.word4 or " ", font=("Segoe UI", 11), command=self.action4)
        if self.hand_detected:
            self.status_label.config(text="● Hand detected", fg="#4ecca3")
        else:
            self.status_label.config(text="● No hand detected", fg="#ff6b6b")

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok or frame is None:
                self.root.after(10, self.video_loop)
                return

            cv2image = cv2.flip(frame, 1)
            cv2image_copy = np.array(cv2image)
            rgb_image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            hands = hd.findHands(cv2image, draw=False, flipType=True)
            self.hand_detected = False

            if hands:
                hand_data = parse_hand(hands[0])
                if hand_data:
                    x, y, w, h = hand_data["bbox"]
                    image = cv2image_copy[y - offset : y + h + offset, x - offset : x + w + offset]
                    if image.size > 0:
                        handz = hd2.findHands(image, draw=False, flipType=True)
                        if handz:
                            inner_hand = parse_hand(handz[0])
                            if inner_hand and inner_hand.get("lmList"):
                                self.hand_detected = True
                                self.pts = inner_hand["lmList"]
                                white = make_white_canvas()
                                skeleton = self.draw_skeleton(white, self.pts, w, h)
                                self.predict(skeleton)

                                skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
                                self.current_image2 = Image.fromarray(skeleton_rgb)
                                imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                                self.panel2.imgtk = imgtk2
                                self.panel2.config(image=imgtk2)

            self.update_ui()
        except Exception:
            print(traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def distance(self,x,y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def _apply_suggestion(self, replacement):
        if not replacement:
            return
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        self.str = self.str[:idx_word] + replacement.upper()
        self.update_ui()

    def action1(self):
        self._apply_suggestion(self.word1)

    def action2(self):
        self._apply_suggestion(self.word2)

    def action3(self):
        self._apply_suggestion(self.word3)

    def action4(self):
        self._apply_suggestion(self.word4)


    def speak_fun(self):
        text = self.str.strip()
        if text:
            self.speak_engine.say(text)
            self.speak_engine.runAndWait()


    def clear_fun(self):
        self.str = ""
        self.word = ""
        self.word1 = ""
        self.word2 = ""
        self.word3 = ""
        self.word4 = ""
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10
        self._debounce_char = None
        self._debounce_count = 0
        self.current_symbol = "-"

    def predict(self, test_image):
        white=test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white, verbose=0)[0], dtype="float32")
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2


        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6


        # condition for [yj][x]
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + 13 < self.pts[8][0] and self.pts[0][0] + 13 < self.pts[12][0] and self.pts[0][0] + 13 < self.pts[16][0] and
                    self.pts[0][0] + 13 < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'

        if ch1 in (1, "E", "S", "X", "Y", "B"):
            if (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] < self.pts[12][1]
                and self.pts[14][1] < self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):
                ch1 = " "

        if ch1 in ("E", "Y", "B"):
            if (self.pts[4][0] < self.pts[5][0]) and (
                self.pts[6][1] > self.pts[8][1]
                and self.pts[10][1] > self.pts[12][1]
                and self.pts[14][1] > self.pts[16][1]
                and self.pts[18][1] > self.pts[20][1]
            ):
                ch1 = "next"

        if ch1 in ("next", "B", "C", "H", "F", "X"):
            if (
                self.pts[0][0] > self.pts[8][0]
                and self.pts[0][0] > self.pts[12][0]
                and self.pts[0][0] > self.pts[16][0]
                and self.pts[0][0] > self.pts[20][0]
            ) and (
                self.pts[4][1] < self.pts[8][1]
                and self.pts[4][1] < self.pts[12][1]
                and self.pts[4][1] < self.pts[16][1]
                and self.pts[4][1] < self.pts[20][1]
            ) and (
                self.pts[4][1] < self.pts[6][1]
                and self.pts[4][1] < self.pts[10][1]
                and self.pts[4][1] < self.pts[14][1]
                and self.pts[4][1] < self.pts[18][1]
            ):
                ch1 = "Backspace"

        # Debounce: only commit gestures after STABLE_FRAMES consistent reads
        if ch1 == self._debounce_char:
            self._debounce_count += 1
        else:
            self._debounce_char = ch1
            self._debounce_count = 1

        if self._debounce_count < STABLE_FRAMES:
            self.current_symbol = ch1 if isinstance(ch1, str) and len(str(ch1)) == 1 else str(ch1)
            return

        stable_ch1 = ch1
        self.current_symbol = stable_ch1 if isinstance(stable_ch1, str) else str(stable_ch1)

        if stable_ch1 == "next" and self.prev_char != "next":
            prev_idx = (self.count - 2) % 10
            committed = self.ten_prev_char[prev_idx]
            if committed == "Backspace":
                self.str = self.str[:-1]
            elif committed not in ("next", "Backspace", " "):
                self.str += committed

        if stable_ch1 == "Backspace" and self.prev_char != "Backspace":
            self.str = self.str[:-1]

        if stable_ch1 == " " and self.prev_char != " ":
            self.str += " "

        self.prev_char = stable_ch1
        self.count += 1
        self.ten_prev_char[self.count % 10] = stable_ch1

        if self.str.strip():
            st = self.str.rfind(" ")
            word = self.str[st + 1 :]
            self.word = word
            if word.strip() and ENCHANT_AVAILABLE:
                suggestions = ddd.suggest(word)
                self.word1 = suggestions[0] if len(suggestions) >= 1 else ""
                self.word2 = suggestions[1] if len(suggestions) >= 2 else ""
                self.word3 = suggestions[2] if len(suggestions) >= 3 else ""
                self.word4 = suggestions[3] if len(suggestions) >= 4 else ""
            else:
                self.word1 = ""
                self.word2 = ""
                self.word3 = ""
                self.word4 = ""


    def destructor(self):
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Application().root.mainloop()
