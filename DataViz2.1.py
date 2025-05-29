import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
import cv2
import os
import matplotlib
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

def gesture_normalize(pose):
    def dis(k1, k2):
        return ((k1[1] - k2[1]) ** 2 + (k1[0] - k2[0]) ** 2) ** 0.5
    unit = dis([pose[11].x, pose[11].y], [pose[23].x, pose[23].y])
    center = [(pose[11].x + pose[12].x) / 2, (pose[11].y + pose[12].y) / 2]
    for i in range(len(pose)):
        pose[i].x = (pose[i].x - center[0]) / unit
        pose[i].y = (pose[i].y - center[1]) / unit
    return pose

class MediaPipeUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MediaPipe 分析器")
        self.video_path = tk.StringVar()
        self.landmark_checkboxes = []
        self.landmark_vars = []

        tk.Label(root, text="影片路徑:").grid(row=0, column=0, sticky='e')
        tk.Entry(root, textvariable=self.video_path, width=40).grid(row=0, column=1)
        tk.Button(root, text="選擇影片", command=self.choose_video).grid(row=0, column=2)

        self.selected_part = tk.StringVar(value="Pose")
        tk.Label(root, text="選擇部位:").grid(row=1, column=0, sticky='e')
        parts = ["Pose", "Hands", "Face"]
        tk.OptionMenu(root, self.selected_part, *parts, command=self.update_landmark_checkboxes).grid(row=1, column=1, sticky='w')

        self.checkbox_frame = tk.LabelFrame(root, text="選擇要畫的點")
        self.checkbox_frame.grid(row=2, column=0, columnspan=3, pady=5)

        tk.Button(root, text="開始分析", command=self.start_analysis).grid(row=3, column=1)

        self.update_landmark_checkboxes("Pose")

        self.models = {
            "Pose": {
                "path": "./model/pose_landmarker_full.task",
                "landmarker": mp.tasks.vision.PoseLandmarker,
                "options": PoseLandmarkerOptions
            },
            "Hands": {
                "path": "./model/hand_landmarker.task",
                "landmarker": mp.tasks.vision.HandLandmarker,
                "options": HandLandmarkerOptions
            },
            "Face": {
                "path": "./model/face_landmarker.task",
                "landmarker": mp.tasks.vision.FaceLandmarker,
                "options": FaceLandmarkerOptions
            }
        }

    def choose_video(self):
        filepath = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if filepath:
            self.video_path.set(filepath)

    def update_landmark_checkboxes(self, part):
        for cb in self.landmark_checkboxes:
            cb.destroy()
        self.landmark_checkboxes.clear()
        self.landmark_vars.clear()
        if part == "Pose":
            count = 33
            offset = 0
        elif part == "Hands":
            for label in ["Left", "Right"]:
                var = tk.BooleanVar()
                cb = tk.Checkbutton(self.checkbox_frame, text=label, variable=var)
                cb.grid(row=0, column=len(self.landmark_vars), sticky='w')
                self.landmark_checkboxes.append(cb)
                self.landmark_vars.append(var)
            count = 21
            offset = 1
        elif part == "Face":
            count = 478
            offset = 0
        else:
            count = 0
            offset = 0

        for i in range(count):
            var = tk.BooleanVar()
            cb = tk.Checkbutton(self.checkbox_frame, text=f"點 {i}", variable=var)
            cb.grid(row=i // 10 + offset, column=i % 10, sticky='w')
            self.landmark_checkboxes.append(cb)
            self.landmark_vars.append(var)

    def start_analysis(self):
        video = self.video_path.get()
        if not os.path.exists(video):
            messagebox.showerror("錯誤", "請選擇一個有效的影片")
            return

        part = self.selected_part.get()
        offset = 2 if part == "Hands" else 0
        selected_ids = [i + offset for i, var in enumerate(self.landmark_vars[offset:]) if var.get()]
        hands_selected = [self.landmark_vars[0].get(), self.landmark_vars[1].get()] if part == "Hands" else [False, False]

        if not selected_ids:
            messagebox.showwarning("提示", "請至少選擇一個點")
            return

        self.mediapipe_analyze(part, video, selected_ids, hands_selected)

    def mediapipe_analyze(self, part, video_path, selected_ids, hands_selected):
        model_info = self.models[part]
        base_options = mp.tasks.BaseOptions(model_asset_path=model_info["path"])
        options = model_info["options"](
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE
        )
        vision_landmarker = model_info["landmarker"]

        cap = cv2.VideoCapture(video_path)

        if part == "Hands":
            data = {
                "Left": {i: {'x': [], 'y': []} for i in selected_ids[2:]},
                "Right": {i: {'x': [], 'y': []} for i in selected_ids[2:]}
            }
        else:
            data = {i: {'x': [], 'y': []} for i in selected_ids}

        with vision_landmarker.create_from_options(options) as landmarker:
            count_image = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                try:
                    result = landmarker.detect(mp_image)
                except Exception as e:
                    print(f"Mediapipe Process Error: {e}")
                    continue

                if part == "Pose":
                    results = result.pose_landmarks
                elif part == "Hands":
                    results = result.hand_landmarks
                elif part == "Face":
                    results = result.face_landmarks
                else:
                    results = []

                if part == "Hands":
                    if hands_selected == [True, True] and len(results) < 2:
                        print(f"Frame {count_image}: 僅偵測到一隻手，跳過紀錄。")
                        count_image += 1
                        continue

                    for hand_idx, (is_selected, label) in enumerate(zip(hands_selected, ["Left", "Right"])):
                        if is_selected and hand_idx < len(results):
                            hand = results[hand_idx]
                            for i in selected_ids[2:]:
                                data[label][i]['x'].append(hand[i - 2].x)
                                data[label][i]['y'].append(hand[i - 2].y)
                        elif is_selected:
                            print(f"Frame {count_image}: {label} 手未被偵測")
                else:
                    if results and len(results) > 0:
                        if part == "Pose":
                            results[0] = gesture_normalize(results[0])
                        for i in selected_ids:
                            data[i]['x'].append(results[0][i].x)
                            data[i]['y'].append(results[0][i].y)
                    else:
                        print(f"Frame {count_image}: 無偵測結果，補 0")
                        for i in selected_ids:
                            data[i]['x'].append(0)
                            data[i]['y'].append(0)

                count_image += 1

        cap.release()

        # 畫圖
        if part == "Hands":
            for label in ["Left", "Right"]:
                if (label == "Left" and hands_selected[0]) or (label == "Right" and hands_selected[1]):
                    for i in selected_ids[2:]:
                        plt.figure()
                        plt.suptitle(f"{label} 手 - 點 {i} 變化圖")
                        plt.subplot(2, 1, 1)
                        plt.plot(data[label][i]['x'])
                        plt.title("X軸變化")
                        plt.subplot(2, 1, 2)
                        plt.plot(data[label][i]['y'])
                        plt.title("Y軸變化")
        else:
            for i in selected_ids:
                plt.figure()
                plt.suptitle(f"點 {i} 變化圖")
                plt.subplot(2, 1, 1)
                plt.plot(data[i]['x'])
                plt.title("X軸變化")
                plt.subplot(2, 1, 2)
                plt.plot(data[i]['y'])
                plt.title("Y軸變化")

        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = MediaPipeUI(root)
    root.mainloop()
