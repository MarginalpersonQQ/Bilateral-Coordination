import tkinter as tk
from tkinter import filedialog, messagebox
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
from mediapipe.framework.formats import landmark_pb2
import cv2
import os
import matplotlib
from multiprocessing import Process
from matplotlib.ticker import MultipleLocator
import cv2
import numpy as np
matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

def draw_landmarks_on_frame(frame, result, part, selected_ids=None):
    h, w, _ = frame.shape

    def draw_selected(landmarks, connections=None):
        for i, lm in enumerate(landmarks.landmark):
            if selected_ids is None or i in selected_ids:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    if part in ["Pose", "Face"]:
        if result:
            draw_selected(result)

    elif part == "Hands":
        if isinstance(result, list):
            for hand_proto in result:
                if hand_proto:
                    draw_selected(hand_proto)

def face_change_center(data):
    base_ids = [33, 133, 263, 362]

    # 確保四個基準點都存在
    if not all(label in data for label in base_ids):
        print("缺少基準點，無法中心化")
        return

    # 取出四個點的 x, y
    base_xs = [np.array(data[label]['x']) for label in base_ids]
    base_ys = [np.array(data[label]['y']) for label in base_ids]

    # 計算每一幀的平均座標（shape: (幀數,)）
    center_x = sum(base_xs) / len(base_xs)
    center_y = sum(base_ys) / len(base_ys)

    # 讓所有點平移：每一幀減掉新原點
    for label in data:
        x_arr = np.array(data[label]['x'])
        y_arr = np.array(data[label]['y'])
        data[label]['x'] = (x_arr - center_x).tolist()
        data[label]['y'] = (y_arr - center_y).tolist()

    return data

def pose_change_center(data):
    # try:
    def dis(k1, k2):  # distance of two point
        d = pow(((k1[1] - k2[1]) * (k1[1] - k2[1]) + (k1[0] - k2[0]) * (k1[0] - k2[0])), .5)
        return d
    #normalization
    for frame in range(len(data)):
        unit = dis([data[11]['x'][frame], data[11]['y'][frame]], [data[23]['x'][frame], data[23]['y'][frame]])
        center = [(data[11]['x'][frame] + data[12]['x'][frame]) / 2, (data[11]['y'][frame] + data[12]['y'][frame]) / 2]
        for point in data.keys():
            data[point]['x'][frame] = (data[point]['x'][frame]-center[0])/unit
            data[point]['y'][frame] = (data[point]['y'][frame]-center[1])/unit
    # except Exception as ex:
    #     print(ex)
    return data

def animate_multiple_landmarks(xy_dict, video_frames, title="Landmark 動畫"):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from multiprocessing import Value
    import cv2
    import numpy as np

    shared_index = Value('i', 0)

    fig, axs = plt.subplots(len(xy_dict), 2, figsize=(8, 4 * len(xy_dict)))
    fig.suptitle(title)

    lines = []

    for i, (label, (x, y)) in enumerate(xy_dict.items()):
        ax_x = axs[i][0] if len(xy_dict) > 1 else axs[0]
        ax_y = axs[i][1] if len(xy_dict) > 1 else axs[1]
        ax_x.set_title(f"{label} X軸變化")
        ax_y.set_title(f"{label} Y軸變化")
        ax_x.set_xlim(0, len(x))
        ax_y.set_xlim(0, len(y))
        margin = 0.01
        # ax_x.set_ylim(np.nanmin(x) - margin, np.nanmax(x) + margin)
        center_x = (np.nanmax(x) + np.nanmin(x)) / 2
        ax_x.set_ylim(center_x - 0.005, center_x + 0.005)
        ax_y.set_ylim(np.nanmin(y) - margin, np.nanmax(y) + margin)
        ax_x.yaxis.set_major_locator(MultipleLocator(0.001))
        ax_y.yaxis.set_major_locator(MultipleLocator(0.002))

        line1, = ax_x.plot([], [], 'r-')
        line2, = ax_y.plot([], [], 'b-')
        lines.append((line1, line2))

    def update(frame):
        idx = shared_index.value % len(next(iter(xy_dict.values()))[0])
        for (x, y), (line1, line2) in zip(xy_dict.values(), lines):
            line1.set_data(range(idx), x[:idx])
            line2.set_data(range(idx), y[:idx])
        return [l for pair in lines for l in pair]

    ani = animation.FuncAnimation(fig, update, frames=len(video_frames), interval=30, blit=True)

    def video_thread():
        i = 0
        while True:
            frame = video_frames[i % len(video_frames)]
            shared_index.value = i % len(video_frames)
            i += 1
            cv2.imshow("Video", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    from threading import Thread
    video_t = Thread(target=video_thread)
    video_t.start()
    plt.show()
    video_t.join()

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
                "Left": {i: {'x': [], 'y': [], 'valid': []} for i in selected_ids},
                "Right": {i: {'x': [], 'y': [], 'valid': []} for i in selected_ids}
            }
        elif part == "Face":
            center_ids = [33, 133, 263, 362]
            all_ids = list(set(selected_ids + center_ids))  # 合併去重
            data = {i: {'x': [], 'y': [], 'valid': []} for i in all_ids}
        else:
            center_ids = [11, 12, 23]
            all_ids = list(set(selected_ids + center_ids))
            data = {i: {'x': [], 'y': [], 'valid': []} for i in all_ids}


        with vision_landmarker.create_from_options(options) as landmarker:
            video_frames = []
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
                    if hands_selected == [True, True] and not all(label in detected_hands for label in ["Left", "Right"]):
                        print(f"Frame {count_image}: 僅偵測到一隻手，補 0")

                    # 分別處理 Left / Right
                    for label, enabled in zip(["Left", "Right"], hands_selected):
                        for idx in data.keys():
                            if enabled and label in detected_hands:
                                hand = detected_hands[label]
                                data[label][idx]['x'].append(hand[idx].x)
                                data[label][idx]['y'].append(hand[idx].y)
                                data[label][idx]['valid'].append(True)
                            elif enabled:
                                data[label][idx]['x'].append(np.nan)
                                data[label][idx]['y'].append(np.nan)
                                data[label][idx]['valid'].append(False)
                elif part == "Face":
                    if results and len(results) > 0:
                        for i in data.keys():
                            data[i]['x'].append(results[0][i].x)
                            data[i]['y'].append(results[0][i].y)
                            data[i]['valid'].append(True)
                    else:
                        print(f"Frame {count_image}: 無偵測結果，補 0")
                        for i in data.keys():
                            data[i]['x'].append(np.nan)
                            data[i]['y'].append(np.nan)
                            data[i]['valid'].append(False)
                else:
                    if results and len(results) > 0:
                        for i in data.keys():
                            data[i]['x'].append(results[0][i].x)
                            data[i]['y'].append(results[0][i].y)
                            data[i]['valid'].append(True)
                    else:
                        print(f"Frame {count_image}: 無偵測結果，補 0")
                        for i in data.keys():
                            data[i]['x'].append(np.nan)
                            data[i]['y'].append(np.nan)
                            data[i]['valid'].append(False)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 拿來畫圖的 landmark
                if part == "Pose":
                    landmark_to_draw = landmark_pb2.NormalizedLandmarkList()
                    for lm in result.pose_landmarks[0]:
                        landmark_to_draw.landmark.add(
                            x=lm.x,
                            y=lm.y,
                            z=lm.z,
                            visibility=lm.visibility,
                            presence=lm.presence
                        )
                elif part == "Hands":
                    landmark_to_draw = []
                    for hand in result.hand_landmarks:
                        proto = landmark_pb2.NormalizedLandmarkList()
                        for lm in hand:
                            proto.landmark.add(
                                x=lm.x,
                                y=lm.y,
                                z=lm.z
                            )
                        landmark_to_draw.append(proto)

                elif part == "Face":
                    landmark_to_draw = landmark_pb2.NormalizedLandmarkList()
                    for lm in result.face_landmarks[0]:
                        landmark_to_draw.landmark.add(x=lm.x, y=lm.y, z=lm.z)
                else:
                    landmark_to_draw = None

                draw_landmarks_on_frame(frame_bgr, landmark_to_draw, part, selected_ids)
                video_frames.append(frame_bgr)
                count_image += 1

        cap.release()

        # 畫圖
        xy_dict = {}

        if part == "Hands":
            for label in ["Left", "Right"]:
                if (label == "Left" and hands_selected[0]) or (label == "Right" and hands_selected[1]):
                    for i in selected_ids:
                        x = data[label][i]['x']
                        y = data[label][i]['y']
                        xy_dict[f"{label} 手 - 點 {i}"] = (x, y)
        elif part == "Face":
            data = face_change_center(data)
            for i in selected_ids:
                x = data[i]['x']
                y = data[i]['y']
                xy_dict[f"點 {i}"] = (x, y)
                print(x, y)
        else:
            data = pose_change_center(data)
            for i in selected_ids:
                x = data[i]['x']
                y = data[i]['y']
                xy_dict[f"點 {i}"] = (x, y)

        p = Process(target=animate_multiple_landmarks, args=(xy_dict, video_frames))
        p.start()
        p.join()

if __name__ == "__main__":
    root = tk.Tk()
    app = MediaPipeUI(root)
    root.mainloop()
