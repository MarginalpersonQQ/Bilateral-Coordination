import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

    def animate_landmark_series(self, x_vals, y_vals, title):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle(title)

        # 動畫線條
        line1, = ax1.plot([], [], 'r-')
        line2, = ax2.plot([], [], 'b-')

        # 設定 X 軸固定（以 frame 數為長度）
        ax1.set_xlim(0, len(x_vals))
        ax2.set_xlim(0, len(y_vals))

        # 設定 Y 軸上下邊界（加點 margin 避免貼邊）
        margin = 0.05
        ax1.set_ylim(min(x_vals) - margin, max(x_vals) + margin)
        ax2.set_ylim(min(y_vals) - margin, max(y_vals) + margin)

        ax1.set_title("X軸變化")
        ax2.set_title("Y軸變化")

        def update(frame):
            line1.set_data(range(frame), x_vals[:frame])
            line2.set_data(range(frame), y_vals[:frame])
            return line1, line2

        ani = animation.FuncAnimation(
            fig, update, frames=len(x_vals), interval=30, blit=True
        )
        plt.show()

    def mediapipe_analyze(self, part, video_path, selected_ids, hands_selected):
        model_info = self.models[part]
        base_options = mp.tasks.BaseOptions(model_asset_path=model_info["path"])
        if part == "Hands":
            options = model_info["options"](
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=2  # 👈 只有 Hands 要加這個參數
            )
        else:
            options = model_info["options"](
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE
            )
        vision_landmarker = model_info["landmarker"]

        cap = cv2.VideoCapture(video_path)

        if part == "Hands":
            data = {
                "Left": {i: {'x': [], 'y': []} for i in selected_ids},
                "Right": {i: {'x': [], 'y': []} for i in selected_ids}# selected_ids[2:]
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
                    results = result
                elif part == "Face":
                    results = result.face_landmarks
                else:
                    results = []

                if part == "Hands":
                    # 建立 label → landmark 對應字典
                    detected_hands = {
                        result.handedness[i][0].category_name: result.hand_landmarks[i]
                        for i in range(len(result.hand_landmarks))
                    }
                    # 如果使用者要求偵測左右手但目前缺其中一手，則跳過整幀
                    if hands_selected == [True, True] and not all(
                            label in detected_hands for label in ["Left", "Right"]):
                        print(f"Frame {count_image}: 僅偵測到一隻手，跳過紀錄。")
                        count_image += 1
                        continue

                    # 分別處理 Left / Right
                    for label, enabled in zip(["Left", "Right"], hands_selected):
                        if enabled:
                            if label in detected_hands:
                                hand = detected_hands[label]
                                for idx in selected_ids:
                                    data[label][idx]['x'].append(hand[idx].x)
                                    data[label][idx]['y'].append(hand[idx].y)
                            else:
                                print(f"Frame {count_image}: {label} 手未被偵測，補 0")
                                for idx in selected_ids:
                                    data[label][idx]['x'].append(0)
                                    data[label][idx]['y'].append(0)

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
                        for i in selected_ids:
                            x_vals = data[label][i]['x']
                            y_vals = data[label][i]['y']
                            self.animate_landmark_series(x_vals, y_vals, f"{label} 手 - 點 {i} 動畫變化")
        else:
            for i in selected_ids:
                x_vals = data[i]['x']
                y_vals = data[i]['y']
                self.animate_landmark_series(x_vals, y_vals, f"點 {i} 動畫變化")

if __name__ == "__main__":
    root = tk.Tk()
    app = MediaPipeUI(root)
    root.mainloop()
