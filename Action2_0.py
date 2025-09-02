import os
from statistics import correlation

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
import matplotlib.pyplot as plt
from pygame.transform import threshold
from scipy.signal import find_peaks, peak_widths
from dtaidistance import dtw

def data_normalize(data):
    # try:
    def dis(k1, k2):  # distance of two point
        d = pow(((k1[1] - k2[1]) * (k1[1] - k2[1]) + (k1[0] - k2[0]) * (k1[0] - k2[0])), .5)
        return d
    #normalization
    for frame in range(len(data)):
        unit = dis([data[frame]['pose'][11]['x'], data[frame]['pose'][11]['y']], [data[frame]['pose'][23]['x'], data[frame]['pose'][23]['y']])
        center = [(data[frame]['pose'][11]['x'] + data[frame]['pose'][12]['x']) / 2, (data[frame]['pose'][11]['y'] + data[frame]['pose'][12]['y']) / 2]
        for type in data[frame].keys():
            if type == 'hand':
                for hand_type in data[frame][type].keys():
                    for point in data[frame][type][hand_type].keys():
                        data[frame][type][hand_type][point]['x'] = (data[frame][type][hand_type][point]['x'] - center[0]) / unit
                        data[frame][type][hand_type][point]['y'] = (data[frame][type][hand_type][point]['y'] - center[1]) / unit
            else:
                for point in data[frame][type].keys():
                    data[frame][type][point]['x'] = (data[frame][type][point]['x']-center[0])/unit
                    data[frame][type][point]['y'] = (data[frame][type][point]['y']-center[1])/unit
    # except Exception as ex:
    #     print(f"ERROR: data_normalize {ex}")
    return data

class MDP:
    def __init__(self):
        self.model_path = r"./model" if os.path.exists(r"./model") else None
        if self.model_path is None:
            print("Model File Not Exist.")

        self.base_options = mp.tasks.BaseOptions
        self.vision_running_mode = mp.tasks.vision.RunningMode

        self.model_config = {
            "pose": {
                "task_file": "pose_landmarker_full.task",
                "landmarker_class": mp.tasks.vision.PoseLandmarker,
                "option_class": PoseLandmarkerOptions,
                "result_key": "pose_landmarks"
            },
            "hand": {
                "task_file": "hand_landmarker.task",
                "landmarker_class": mp.tasks.vision.HandLandmarker,
                "option_class": HandLandmarkerOptions,
                "result_key": "hand_landmarks"
            },
            "face": {
                "task_file": "face_landmarker.task",
                "landmarker_class": mp.tasks.vision.FaceLandmarker,
                "option_class": FaceLandmarkerOptions,
                "result_key": "face_landmarks"
            }
        }

        self.landmarkers = {}
        self.init_all_models()
        print("Mediapipe Initialized\n")

    @staticmethod
    def posepoint(x, y):
        return {"x": x, "y": y}

    def init_all_models(self):
        for model_type, config in self.model_config.items():
            task_path = os.path.join(self.model_path, config["task_file"])
            if not os.path.exists(task_path):
                print(f"[Warning] Model file not found for {model_type}: {task_path}")
                continue
            try:
                if model_type == "hand":
                    options = config["option_class"](
                        base_options=self.base_options(model_asset_path=task_path),
                        running_mode=self.vision_running_mode.IMAGE,
                        num_hands=2,
                        )
                else:
                    options = config["option_class"](
                        base_options=self.base_options(model_asset_path=task_path),
                        running_mode=self.vision_running_mode.IMAGE)
                self.landmarkers[model_type] = config["landmarker_class"].create_from_options(options)
            except Exception as e:
                print(f"[Error] Failed to initialize {model_type}: {e}")

    def  _process_video(self, video_path, use_models):
        cap = cv2.VideoCapture(video_path)
        data = {}
        count_image = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"\033[94mMessage:  No Frame In Here.\033[0m")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)  # 使用 create_from_array 比較穩定
                data[count_image] = {}

                for model_type in use_models:
                    if model_type not in self.landmarkers:
                        print(f"[Skip] Model '{model_type}' not initialized.")
                        continue

                    landmarker = self.landmarkers[model_type]
                    result = landmarker.detect(mp_image)
                    key = self.model_config[model_type]["result_key"]
                    result_data = getattr(result, key, [])

                    if model_type == "hand":
                        # 處理 hand (left / right)
                        detected_hands = {
                            result.handedness[i][0].category_name: result.hand_landmarks[i]
                            for i in range(len(result.hand_landmarks))
                        }
                        data[count_image]["hand"] = {"left": {}, "right": {}}
                        for hand_label in ["Left", "Right"]:
                            if hand_label in detected_hands:
                                hand_landmarks = detected_hands[hand_label]
                                data[count_image]["hand"][hand_label.lower()] = {
                                    i: self.posepoint(lm.x, lm.y)
                                    for i, lm in enumerate(hand_landmarks)
                                }
                            else:
                                data[count_image]["hand"][hand_label.lower()] = {
                                    i: self.posepoint(float("nan"), float("nan"))
                                    for i in range(21)  # hand 有 21 個點
                                }
                    else:
                        if result_data and len(result_data) > 0 and result_data[0]:
                            data[count_image][model_type] = {
                                i: self.posepoint(lm.x, lm.y)
                                for i, lm in enumerate(result_data[0])
                            }
                        else:
                            print(f"[{model_type}] No landmark at frame {count_image}, copying previous.")
                            data[count_image][model_type] = data.get(count_image - 1, {}).get(model_type, {})
                count_image += 1
        except Exception as E:
            print(f"\033[93mexception: {E}\033[0m")
        finally:
            print(f"Mediapipe processed images : {count_image}")
            cap.release()
        return data

    def get_data(self, video_path, models=("pose", "hand", "face")):
        """
        models: tuple of model types to use, e.g., ("pose", "hand")
        """
        return self._process_video(video_path, models)

    def close(self):
        for landmarker in self.landmarkers.values():
            landmarker.close()
        self.landmarkers.clear()

class PeakDataStruct:
    def __init__(self, start=0.0, start_pos=0, end=0.0, end_pos=0, peak_max=0.0, peak_max_pos=0, peak_width = 0):
        self.start = start
        self.start_pos = start_pos
        self.end = end
        self.end_pos = end_pos
        self.peak_max = peak_max
        self.peak_max_pos = peak_max_pos
        self.peak_width = peak_width
        self.count_score_peak_id = None

class PalmOrientationDecide:
    @staticmethod
    def avg_abs_process(data): # 平均值原點
        data_mean = np.nanmean(data)
        data_diff = np.abs(data - data_mean)
        return data_diff

    @staticmethod
    def threshold_filter(data): # 數值過濾
        std_threshold = 1.25
        threshold = np.nanmean(data) + np.nanstd(data) * std_threshold
        outlier_mask = data > threshold
        adjusted_data = data.copy()
        adjusted_data[outlier_mask] = np.nan
        return [adjusted_data, threshold]

    @staticmethod
    def linear_interpolate(data): # 線性插值
        nans = np.isnan(data)
        x = np.arange(len(data))
        data[nans] = np.interp(x[nans], x[~nans], data[~nans])
        return data

    @staticmethod
    def palm_orientation(data): # 差距計算
        def count_delta(data):
            delta = []
            for x1, x2 in zip(data[0], data[1]):
                # print(f"4: {x1}, 20:{x2}")
                delta.append(x1 - x2)
            delta = np.array(delta)
            return delta

        return count_delta(data)

class LandmarkDataProcess:
    @staticmethod
    def find_peak(data, config, prominence = 0.3, pose_axis="x", hand_axis="x"):
        processed_data = {}
        for model_type in config.keys():
            processed_data[model_type] = {}
        result = {}
        data = data_normalize(data)
        for model_type in config.keys():
            if model_type == "pose":
                result["pose"] = {}
                print(f"find_pose_peak")
                target = np.array(
                    [np.array([data[frame + 1]['pose'][point][pose_axis] - data[frame]['pose'][point][pose_axis] for frame in range(len(data) - 1)]) for point in
                     config[model_type]])

                for i, point in enumerate(config[model_type]):
                    target[i] = PalmOrientationDecide.linear_interpolate(target[i]) # 共用手部補值function
                    processed_data[model_type][point] = target[i]
                    peaks, peak_height = find_peaks(target[i], height=target[i].mean(), distance=10, prominence=prominence)
                    widths, heights, left_ips, right_ips = peak_widths(target[i], peaks)
                    temp = []
                    for p in range(len(peaks)):
                        t = PeakDataStruct()
                        t.peak_max_pos = peaks[p]
                        t.peak_max = peak_height['peak_heights'][p]
                        t.peak_width = widths[p]
                        t.start = t.end = heights[p]
                        t.start_pos = left_ips[p]
                        t.end_pos = right_ips[p]
                        t.count_score_peak_id = point
                        temp.append(t)
                    result['pose'][point] = temp
                for data in target:
                    plt.plot(data)
                    plt.show()
            elif model_type == "hand":
                print(f"find_hand_peak")
                temp = {}
                for which_hand in ["left", "right"]:
                    target = np.array( # 數據提取 左右移動
                        [np.array([data[frame]['hand'][which_hand][point][hand_axis] for frame in range(len(data))]) for point in
                         config[model_type]])
                    temp[which_hand] = [] #手部數據字典初始化
                    plt.plot(target[0])
                    plt.plot(target[1])
                    plt.xlim(0, len(data))
                    plt.show()
                    for i, point in enumerate(config[model_type]):
                        """手部數據遺漏點處理"""
                        target[i] = PalmOrientationDecide.avg_abs_process(target[i])
                        target[i], th = PalmOrientationDecide.threshold_filter(target[i])
                        target[i] = PalmOrientationDecide.linear_interpolate(target[i])
                        processed_data[model_type][which_hand] = target[i].copy()
                    """手部差距計算"""
                    delta = PalmOrientationDecide.palm_orientation([target[0], target[1]])
                    peaks, peak_height = find_peaks(delta, height=delta.mean(), distance=20, prominence=0.2)
                    widths, heights, left_ips, right_ips = peak_widths(delta, peaks)
                    # print(len(peaks))
                    for p in range(len(peaks)):
                        t = PeakDataStruct()
                        t.peak_max_pos = peaks[p]
                        t.peak_max = peak_height['peak_heights'][p]
                        t.peak_width = widths[p]
                        t.start = t.end = heights[p]
                        t.start_pos = left_ips[p]
                        t.end_pos = right_ips[p]
                        t.count_score_peak_id = which_hand
                        temp[which_hand].append(t)
                result["hand"] = temp
            elif model_type == "face":
                pass
        return [result, processed_data]

    @staticmethod
    def flip_data_around_mean(data):
        temp = {}
        result = {}

        for model_type in data[0].keys():
            for point in data[0][model_type].keys(): # 建立暫存容器
                temp[point] = {'x':[], 'y':[]}
                result[point] = {'x':[], 'y':[]}

            for frame in range(len(data)): # 提取資料填入容器
                for point in data[frame][model_type].keys():
                    for axis in data[frame][model_type][point].keys():
                        temp[point][axis].append(data[frame][model_type][point][axis])

            for point in temp.keys(): # 反轉資料
                for axis in temp[point].keys():
                    temp[point][axis] = np.array(temp[point][axis])
                    mean = temp[point][axis].mean()
                    result[point][axis] = 2 * mean - temp[point][axis]

            for frame in range(len(data)): # 儲存容器回資料
                for point in data[frame][model_type].keys():
                    for axis in data[frame][model_type][point].keys():
                        data[frame][model_type][point][axis] = result[point][axis][frame]

        return data

    @staticmethod
    def flexible_pattern_match(actual_order, expected_patterns, penalty_per_extra=10):
        best_score = 0

        for expected_pattern in expected_patterns:
            L = len(expected_pattern)
            for start in range(len(actual_order) - 1):  # 可變起始點
                for end in range(start + 1, len(actual_order) + 1):  # 可變終止點
                    window = actual_order[start:end]
                    if len(window) > L:
                        continue  # 超過 pattern 長度不比較（可改為允許部分比對）

                    match = sum(1 for a, e in zip(window, expected_pattern) if a == e)
                    match_score = (match / L) * 100
                    used_len = len(window)
                    total_len = len(actual_order)

                    # 干擾 = 除了這段之外的點（前後）
                    num_noise = total_len - used_len
                    penalty = num_noise * penalty_per_extra

                    final_score = max(0, match_score - penalty)
                    best_score = max(best_score, final_score)

        return best_score

    @staticmethod
    def two_data_correlation(data, data_info, point):
        # plt.plot(data[point[0]])
        # plt.plot(data[point[1]])
        # plt.show()
        slice_0 = data[point[0]][int(data_info[point[0]]["start"]):int(data_info[point[0]]["end"]) + 1]
        slice_1 = data[point[1]][int(data_info[point[1]]["start"]):int(data_info[point[1]]["end"]) + 1]
        min_len = min(len(slice_0), len(slice_1))
        slice_0 = slice_0[:min_len]
        slice_1 = slice_1[:min_len]
        corr = np.corrcoef(np.array(slice_0), np.array(slice_1))[0, 1]
        return corr

class Action1:
    def __init__(self, path):
        self.config = {'pose':[15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, peak_data, processed_data):
        data = []
        peak_width = [] # 波的寬度
        peak_height = []
        two_peak_maximum_pos_gap = [] # 兩波峰的距離
        movement_start_end_pos = {}
        two_peak_distance = [] # 前一個波的結束到下一個波的開始的距離(Not Use)
        st_to_max_to_end_diff = [] #下去 -> 上來的時間差距
        peak_pattern = []
        for point in self.config['pose']:
            movement_start_end_pos[point] = {"start" : peak_data['pose'][point][0].start_pos, "end" : peak_data['pose'][point][-1].end_pos}
            for i in range(len(peak_data['pose'][point])):
                data.append(peak_data['pose'][point][i])
        data.sort(key = lambda x : x.peak_max_pos) # sort by peak maximum position

        for i in range(len(data)):
            peak_pattern.append(data[i].count_score_peak_id)
            peak_width.append(data[i].end_pos - data[i].start_pos)
            peak_height.append(data[i].peak_max)
            st_to_max_to_end_diff.append(abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"Action1\n")
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")  # 目前沒使用
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        #region score judgement
        expected_patterns = [[15, 16, 15, 16 ,15, 16], [16, 15, 16, 15, 16, 15]]
        # 第一個判斷 正確性 (順序)
        score_0 = LandmarkDataProcess.flexible_pattern_match(peak_pattern, expected_patterns)
        self.score[0] = score_0
        #第二個判斷 左右協調性(左右相關係數)
        score_1 = LandmarkDataProcess.two_data_correlation(processed_data['pose'], movement_start_end_pos, [15, 16])* 100
        self.score[1] = score_1
        #第三個判斷 時間流暢性(長度)(動作單位 : 換手為一個單位)
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        peak_width_std = peak_width.std()
        peak_width_cv = peak_width_std / peak_width_mean
        score_2 = (1 - peak_width_cv) * 100
        self.score[2] = score_2
        #第四個判斷 空間流暢性(幅度)(動作單位 : 換手為一個單位)
        peak_height = np.array(peak_height)
        peak_height_mean = peak_height.mean()
        peak_height_std = peak_height.std()
        peak_height_cv = peak_height_std / peak_height_mean
        score_3 = (1 - peak_height_cv) * 100
        self.score[3] = score_3
        print(f"Action1 1: {score_0}, 2: {score_1}, 3: {score_2}, 4:{score_3}")
        #endregion

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        peak_data, processed_data = LandmarkDataProcess.find_peak(raw_data, self.config, pose_axis="y")
        self.count_score(peak_data, processed_data)

class Action2:
    def __init__(self, path):
        self.config = {'pose': [15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, peak_data, processed_data):

        data = []
        peak_width = []  # 波的寬度
        peak_height = []
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        movement_start_end_pos = {}
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離(Not Use)
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        peak_pattern = []
        for point in self.config['pose']:
            movement_start_end_pos[point] = {"start": peak_data['pose'][point][0].start_pos,
                                             "end": peak_data['pose'][point][-1].end_pos}
            for i in range(len(peak_data['pose'][point])):
                data.append(peak_data['pose'][point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position

        for i in range(len(data)):
            peak_pattern.append(data[i].count_score_peak_id)
            peak_width.append(data[i].end_pos - data[i].start_pos)
            peak_height.append(data[i].peak_max)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"Action1\n")
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")  # 目前沒使用
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # region score judgement
        expected_patterns = [[15, 15, 16, 16, 15, 15, 16, 16], [16, 16, 15, 15, 16, 16, 15, 15]]
        # 第一個判斷 正確性 (順序)
        score_0 = LandmarkDataProcess.flexible_pattern_match(peak_pattern, expected_patterns)
        self.score[0] = score_0
        # 第二個判斷 左右協調性(左右相關係數)
        score_1 = LandmarkDataProcess.two_data_correlation(processed_data['pose'], movement_start_end_pos,
                                                           [15, 16]) * 100
        self.score[1] = score_1
        # 第三個判斷 時間流暢性(長度)(動作單位 : 換手為一個單位)
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        peak_width_std = peak_width.std()
        peak_width_cv = peak_width_std / peak_width_mean
        score_2 = (1 - peak_width_cv) * 100
        self.score[2] = score_2
        # 第四個判斷 空間流暢性(幅度)(動作單位 : 換手為一個單位)
        peak_height = np.array(peak_height)
        peak_height_mean = peak_height.mean()
        peak_height_std = peak_height.std()
        peak_height_cv = peak_height_std / peak_height_mean
        score_3 = (1 - peak_height_cv) * 100
        self.score[3] = score_3
        print(f"Action2 1: {score_0}, 2: {score_1}, 3: {score_2}, 4:{score_3}")
        # endregion
        #endregion

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        peak_data, processed_data = LandmarkDataProcess.find_peak(raw_data, self.config, pose_axis = 'y')
        self.count_score(peak_data, processed_data)

class Action3:
    def __init__(self, path):
        self.config = {'pose': [15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, peak_data, processed_data):
        data = []
        peak_width = []  # 波的寬度
        peak_height = []
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        movement_start_end_pos = {}
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離(Not Use)
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        peak_pattern = []
        for point in self.config['pose']:
            movement_start_end_pos[point] = {"start": peak_data['pose'][point][0].start_pos,
                                             "end": peak_data['pose'][point][-1].end_pos}
            for i in range(len(peak_data['pose'][point])):
                data.append(peak_data['pose'][point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position

        for i in range(len(data)):
            peak_pattern.append(data[i].count_score_peak_id)
            peak_width.append(data[i].end_pos - data[i].start_pos)
            peak_height.append(data[i].peak_max)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"Action1\n")
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")  # 目前沒使用
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # region score judgement
        expected_patterns = [[15, 16, 16, 16, 15, 15], [16, 15, 15, 15, 16, 16]]
        # 第一個判斷 正確性 (順序)
        score_0 = LandmarkDataProcess.flexible_pattern_match(peak_pattern, expected_patterns)
        self.score[0] = score_0
        # 第二個判斷 左右協調性(左右相關係數)(這個動作不適用)
        score_1 = LandmarkDataProcess.two_data_correlation(processed_data['pose'], movement_start_end_pos,[15, 16]) * 100
        self.score[1] = score_1
        # 第三個判斷 時間流暢性(長度)(動作單位 : 換手為一個單位)
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        peak_width_std = peak_width.std()
        peak_width_cv = peak_width_std / peak_width_mean
        score_2 = (1 - peak_width_cv) * 100
        self.score[2] = score_2
        # 第四個判斷 空間流暢性(幅度)(動作單位 : 換手為一個單位)
        peak_height = np.array(peak_height)
        peak_height_mean = peak_height.mean()
        peak_height_std = peak_height.std()
        peak_height_cv = peak_height_std / peak_height_mean
        score_3 = (1 - peak_height_cv) * 100
        self.score[3] = score_3
        print(f"Action3 1: {score_0}, 2: {score_1}, 3: {score_2}, 4:{score_3}")
        # endregion
        # endregion

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        peak_data, processed_data = LandmarkDataProcess.find_peak(raw_data, self.config, pose_axis = 'y')
        self.count_score(peak_data, processed_data)

class Action4:
    def __init__(self, path):
        self.config = {'pose': [15, 16], 'hand' : [4, 20]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, peak_data, processed_data):
        #region pose score count
        data = []
        peak_width = []  # 波的寬度(動作時長)
        peak_height = []  # 波的高度(動作震幅)
        two_peak_maximum_pos_gap = []  # 兩波峰的距離(無用)
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離(無用)
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距(無用)
        for i in range(len(peak_data['pose'])):
            for key in peak_data['pose'].keys():
                for peak in peak_data['pose'][key]:
                    peak_width.append(peak.peak_width)
                    peak_height.append(peak.peak_max)

        # region score judgement
        # 第一個判斷 拍12下
        num_of_peak = 12
        score_1 = 100 - (abs(num_of_peak - (len(peak_data["pose"][15]) + len(peak_data["pose"][16]))) * 100 / num_of_peak)
        self.score[0] = score_1
        # 第二個判斷 振幅的一致性 相同的動作 不管左右手 用標準差的公式
        left = np.array(processed_data['pose'][15])
        right = np.array(processed_data['pose'][16])
        corr_matrix = np.corrcoef(left, right)
        correlation  = corr_matrix[0, 1]
        score_2 =  correlation * 100
        self.score[1] = score_2
        # 第三個判斷 動作時間的一致性 相同的動作 看整個動作的時間 不管左右手 用標準差的公式
        width = np.array(peak_width)
        width_mean = width.mean()
        width_std = width.std()
        width_cv = width_std / width_mean
        score_3 = (1 - width_cv) * 100
        self.score[2] = score_3
        # 第四個判斷 拍一下的流暢度 計算相關係數
        height = np.array(peak_height)
        height_mean = height.mean()
        height_std = height.std()
        height_cv = height_std / height_mean
        score_4 = (1 - height_cv) * 100
        self.score[3] = score_4
        # 顯示輸出
        print(f"Action4 pose 1: {score_1}, 2: {score_2}, 3: {score_3}, 4:{score_4}")
        # endregion
        # endregion

        # region hand score count
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        for i in range(len(peak_data['hand'])):
            for key in peak_data['hand'].keys():
                for peak in peak_data['hand'][key]:
                    peak_width.append(peak.peak_width)
                    peak_height.append(peak.peak_max)

        # region score judgement
        # 第一個判斷
        num_of_peak = 6
        score_0 = 100 - (abs(num_of_peak - (len(peak_data["hand"]["left"]) + len(peak_data["hand"]["right"]))) * 100 / num_of_peak)
        self.score[0] = self.score[0] * 0.7 + score_0 * 0.3
        # 第二個判斷
        left = np.array(processed_data["hand"]["left"])
        right = np.array(processed_data["hand"]["right"])
        corr_matrix = np.corrcoef(left, right)
        correlation = corr_matrix[0, 1]
        score_1 = 100 if correlation * 100 > 70 else 60
        self.score[1] = self.score[1] * 0.7 + score_1 * 0.3
        # 第三個判斷 動作時間的一致性 相同的動作 看整個動作的時間 不管左右手 用標準差的公式
        width = np.array(peak_width)
        width_mean = width.mean()
        width_std = width.std()
        width_cv = width_std / width_mean
        score_2 = (1 - width_cv) * 100
        self.score[2] = self.score[2] * 0.7 + score_2 * 0.3
        # 第四個判斷 拍一下的流暢度 計算相關係數
        height = np.array(peak_height)
        height_mean = height.mean()
        height_std = height.std()
        height_cv = height_std / height_mean
        score_3 = (1 - height_cv) * 100
        self.score[3] = self.score[3]
        # 顯示輸出
        print(f"Action4 hand 1: {score_0}, 2: {score_1}, 3: {score_2}, 4:{score_3}")
        # endregion
        # endregion

        print(f"Action4 1: {self.score[0]}, 2: {self.score[1]}, 3: {self.score[2]}, 4:{self.score[3]}")

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data, processed_data= LandmarkDataProcess.find_peak(raw_data, self.config, pose_axis = 'y', hand_axis = 'x')
        self.count_score(data, processed_data)

class Action5:
    def __init__(self, path):
        self.config = {'pose': [11, 12, 15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, result_data):
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        num_of_peak = 4
        for point in self.config['pose']:
            for i in range(len(result_data['pose'][point])):
                data.append(result_data['pose'][point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position
        for i in range(len(data)):
            peak_width.append(data[i].end_pos - data[i].start_pos)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)

        # region score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        score_1 = 100 - (abs(num_of_peak - len(data)) * 100 / num_of_peak)
        self.score[0] = score_1
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越AW小越好)
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        peak_width_std = peak_width.std()
        peak_width_cv = peak_width_std / peak_width_mean
        score_2 = (1 - peak_width_cv) * 100
        self.score[1] = score_2
        # 第三個判斷 拍一下間隔的時間(差距盡量要相同 越連續)
        two_peak_maximum_pos_gap = np.array(two_peak_maximum_pos_gap)
        two_peak_maximum_pos_gap_mean = two_peak_maximum_pos_gap.mean()
        two_peak_maximum_pos_gap_std = two_peak_maximum_pos_gap.std()
        score_3 = (1 - two_peak_maximum_pos_gap_std / two_peak_maximum_pos_gap_mean) * 100
        self.score[2] = score_3
        # 第四個判斷 拍一下的流暢度(拍下去 與回到初始位置的時間)
        st_to_max_to_end_diff = np.array(st_to_max_to_end_diff)
        st_to_max_to_end_diff_mean = st_to_max_to_end_diff.mean()
        st_to_max_to_end_diff_std = st_to_max_to_end_diff.std()
        score_4 = (1 - st_to_max_to_end_diff_std / st_to_max_to_end_diff_mean) * 100
        self.score[3] = score_4
        # 顯示輸出
        print(f"Action5 1: {score_1}, 2: {score_2}, 3: {score_3}, 4:{score_4}")
        # endregion

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        peak_data_x, processed_raw_data_pose_x = LandmarkDataProcess.find_peak(raw_data, self.config, prominence = 0.1, pose_axis = "x")
        peak_data_y, processed_raw_data_pose_y = LandmarkDataProcess.find_peak(raw_data, self.config, prominence = 0.1, pose_axis = "y")
        rev_raw_data = LandmarkDataProcess.flip_data_around_mean(raw_data.copy())
        peak_data_rev_x, processed_rev_raw_data_pose_x = LandmarkDataProcess.find_peak(rev_raw_data, self.config, prominence = 0.15, pose_axis = "x")
        print(len(peak_data_x["pose"][15]))
        print(len(peak_data_rev_x["pose"][15]))

        # plt.subplot(1, 2, 1)
        # y_data = processed_raw_data_pose_x['pose'][15]
        # plt.plot(y_data, color="green")
        # plt.plot([y_data.mean()] * len(y_data), color="green")
        # peak_list = peak_data_x['pose'][15]
        # peak_positions = [p.peak_max_pos for p in peak_list]
        # peak_values = [y_data[int(p.peak_max_pos)] for p in peak_list]
        # plt.scatter(peak_positions, peak_values, color='red', label='peaks', zorder=5)
        #
        # plt.subplot(1, 2, 2)
        # y_data_rev = processed_rev_raw_data_pose_x['pose'][15]
        # plt.plot(y_data_rev)
        # plt.plot([y_data_rev.mean()] * len(y_data_rev), color="green")
        # peak_list_rev = peak_data_rev_x['pose'][15]
        # peak_positions_rev = [p.peak_max_pos for p in peak_list_rev]
        # peak_values_rev = [y_data_rev[int(p.peak_max_pos)] for p in peak_list_rev]
        # plt.scatter(peak_positions_rev, peak_values_rev, color='red', label='peaks', zorder=5)
        # plt.show()
        # self.count_score(datda)

class Action6:
    def __init__(self, path):
        self.config = {'pose': [0, 11, 12, 13, 14, 15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, peak_data, processed_data):
        data = []
        peak_width = []  # 波的寬度
        peak_height = []
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        movement_start_end_pos = {}
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離(Not Use)
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        peak_pattern = []
        for point in self.config['pose']:
            movement_start_end_pos[point] = {"start": peak_data['pose'][point][0].start_pos,
                                             "end": peak_data['pose'][point][-1].end_pos}
            for i in range(len(peak_data['pose'][point])):
                data.append(peak_data['pose'][point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position

        for i in range(len(data)):
            peak_pattern.append(data[i].count_score_peak_id)
            peak_width.append(data[i].end_pos - data[i].start_pos)
            peak_height.append(data[i].peak_max)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"Action1\n")
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")  # 目前沒使用
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # region score judgement
        expected_patterns = [[15, 16, 16, 16, 15, 15], [16, 15, 15, 15, 16, 16]]
        # 第一個判斷 正確性 (順序)
        score_0 = LandmarkDataProcess.flexible_pattern_match(peak_pattern, expected_patterns)
        self.score[0] = score_0
        # 第二個判斷 左右協調性(左右相關係數)(這個動作不適用)
        score_1 = LandmarkDataProcess.two_data_correlation(processed_data['pose'], movement_start_end_pos,
                                                           [15, 16]) * 100
        self.score[1] = score_1
        # 第三個判斷 時間流暢性(長度)(動作單位 : 換手為一個單位)
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        peak_width_std = peak_width.std()
        peak_width_cv = peak_width_std / peak_width_mean
        score_2 = (1 - peak_width_cv) * 100
        self.score[2] = score_2
        # 第四個判斷 空間流暢性(幅度)(動作單位 : 換手為一個單位)
        peak_height = np.array(peak_height)
        peak_height_mean = peak_height.mean()
        peak_height_std = peak_height.std()
        peak_height_cv = peak_height_std / peak_height_mean
        score_3 = (1 - peak_height_cv) * 100
        self.score[3] = score_3
        print(f"Action3 1: {score_0}, 2: {score_1}, 3: {score_2}, 4:{score_3}")
        # endregion
        # endregion

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        peak_data, processed_data = LandmarkDataProcess.find_peak(raw_data, self.config)
        self.count_score(peak_data)

class Action7:
    def __init__(self, path):
        self.config = {'pose': [11, 12, 13, 14, 15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, raw_data):
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.config['pose']:
            for i in range(len(raw_data[point])):
                data.append(raw_data[point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position
        for i in range(len(data)):
            peak_width.append(data[i].end_pos - data[i].start_pos)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 12:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 5
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        temp_sc = 13
        for i in range(len(peak_width)):
            if abs(peak_width[i] - peak_width_mean) > 2:
                temp_sc -= 2
            elif abs(peak_width[i] - peak_width_mean) > 1:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[1] = temp_sc
        score += temp_sc
        # 第三個判斷 拍一下間隔的時間(差距盡量要相同 越連續) 共13分
        two_peak_maximum_pos_gap = np.array(two_peak_maximum_pos_gap)
        two_peak_maximum_pos_gap_mean = two_peak_maximum_pos_gap.mean()
        temp_sc = 13
        for i in range(len(two_peak_maximum_pos_gap)):
            if abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 3:
                temp_sc -= 2
            elif abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 2:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[2] = temp_sc
        score += temp_sc
        # 第四個判斷 拍一下的流暢度(拍下去 與回到初始位置的時間) 共13分
        st_to_max_to_end_diff = np.array(st_to_max_to_end_diff)
        temp_sc = 13
        for i in range(len(st_to_max_to_end_diff)):
            if abs(st_to_max_to_end_diff[i]) > 6:
                temp_sc -= 2
            elif abs(st_to_max_to_end_diff[i]) > 5:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[3] = temp_sc
        score += temp_sc
        print(f"score: {score}")
        if score >= 80:
            print(f"很棒")
        elif score >= 70:
            print(f"普通")
        else:
            print(f"很差")

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = LandmarkDataProcess.find_peak(raw_data, self.config)
        self.count_score(data)

class Action8:
    def __init__(self, path):
        self.config = {'pose': [11, 12, 13, 14, 15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, raw_data):
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.config['pose']:
            for i in range(len(raw_data[point])):
                data.append(raw_data[point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position
        for i in range(len(data)):
            peak_width.append(data[i].end_pos - data[i].start_pos)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 12:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 5
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        temp_sc = 13
        for i in range(len(peak_width)):
            if abs(peak_width[i] - peak_width_mean) > 2:
                temp_sc -= 2
            elif abs(peak_width[i] - peak_width_mean) > 1:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[1] = temp_sc
        score += temp_sc
        # 第三個判斷 拍一下間隔的時間(差距盡量要相同 越連續) 共13分
        two_peak_maximum_pos_gap = np.array(two_peak_maximum_pos_gap)
        two_peak_maximum_pos_gap_mean = two_peak_maximum_pos_gap.mean()
        temp_sc = 13
        for i in range(len(two_peak_maximum_pos_gap)):
            if abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 3:
                temp_sc -= 2
            elif abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 2:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[2] = temp_sc
        score += temp_sc
        # 第四個判斷 拍一下的流暢度(拍下去 與回到初始位置的時間) 共13分
        st_to_max_to_end_diff = np.array(st_to_max_to_end_diff)
        temp_sc = 13
        for i in range(len(st_to_max_to_end_diff)):
            if abs(st_to_max_to_end_diff[i]) > 6:
                temp_sc -= 2
            elif abs(st_to_max_to_end_diff[i]) > 5:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[3] = temp_sc
        score += temp_sc
        print(f"score: {score}")
        if score >= 80:
            print(f"很棒")
        elif score >= 70:
            print(f"普通")
        else:
            print(f"很差")

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = LandmarkDataProcess.find_peak(raw_data, self.config)
        self.count_score(data)

class Action9:
    def __init__(self, path):
        self.config = {'hand' : [4, 8, 12, 16, 20]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, raw_data):
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.config['pose']:
            for i in range(len(raw_data[point])):
                data.append(raw_data[point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position
        for i in range(len(data)):
            peak_width.append(data[i].end_pos - data[i].start_pos)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 12:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 5
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        temp_sc = 13
        for i in range(len(peak_width)):
            if abs(peak_width[i] - peak_width_mean) > 2:
                temp_sc -= 2
            elif abs(peak_width[i] - peak_width_mean) > 1:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[1] = temp_sc
        score += temp_sc
        # 第三個判斷 拍一下間隔的時間(差距盡量要相同 越連續) 共13分
        two_peak_maximum_pos_gap = np.array(two_peak_maximum_pos_gap)
        two_peak_maximum_pos_gap_mean = two_peak_maximum_pos_gap.mean()
        temp_sc = 13
        for i in range(len(two_peak_maximum_pos_gap)):
            if abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 3:
                temp_sc -= 2
            elif abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 2:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[2] = temp_sc
        score += temp_sc
        # 第四個判斷 拍一下的流暢度(拍下去 與回到初始位置的時間) 共13分
        st_to_max_to_end_diff = np.array(st_to_max_to_end_diff)
        temp_sc = 13
        for i in range(len(st_to_max_to_end_diff)):
            if abs(st_to_max_to_end_diff[i]) > 6:
                temp_sc -= 2
            elif abs(st_to_max_to_end_diff[i]) > 5:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[3] = temp_sc
        score += temp_sc
        print(f"score: {score}")
        if score >= 80:
            print(f"很棒")
        elif score >= 70:
            print(f"普通")
        else:
            print(f"很差")

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = LandmarkDataProcess.find_peak(raw_data, self.config)

        self.count_score(data)

class Action10:
    def __init__(self, path):
        self.config = {'pose': [23, 24, 25, 26, 27, 28]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, raw_data):
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.config['pose']:
            for i in range(len(raw_data[point])):
                data.append(raw_data[point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position
        for i in range(len(data)):
            peak_width.append(data[i].end_pos - data[i].start_pos)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 12:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 5
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        temp_sc = 13
        for i in range(len(peak_width)):
            if abs(peak_width[i] - peak_width_mean) > 2:
                temp_sc -= 2
            elif abs(peak_width[i] - peak_width_mean) > 1:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[1] = temp_sc
        score += temp_sc
        # 第三個判斷 拍一下間隔的時間(差距盡量要相同 越連續) 共13分
        two_peak_maximum_pos_gap = np.array(two_peak_maximum_pos_gap)
        two_peak_maximum_pos_gap_mean = two_peak_maximum_pos_gap.mean()
        temp_sc = 13
        for i in range(len(two_peak_maximum_pos_gap)):
            if abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 3:
                temp_sc -= 2
            elif abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 2:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[2] = temp_sc
        score += temp_sc
        # 第四個判斷 拍一下的流暢度(拍下去 與回到初始位置的時間) 共13分
        st_to_max_to_end_diff = np.array(st_to_max_to_end_diff)
        temp_sc = 13
        for i in range(len(st_to_max_to_end_diff)):
            if abs(st_to_max_to_end_diff[i]) > 6:
                temp_sc -= 2
            elif abs(st_to_max_to_end_diff[i]) > 5:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[3] = temp_sc
        score += temp_sc
        print(f"score: {score}")
        if score >= 80:
            print(f"很棒")
        elif score >= 70:
            print(f"普通")
        else:
            print(f"很差")

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = LandmarkDataProcess.find_peak(raw_data, self.config)
        self.count_score(data)

class Action11:
    def __init__(self, path):
        self.config = {'pose': [23, 24, 25, 26, 27, 28]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, raw_data):
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.config['pose']:
            for i in range(len(raw_data[point])):
                data.append(raw_data[point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position
        for i in range(len(data)):
            peak_width.append(data[i].end_pos - data[i].start_pos)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 12:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 5
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        temp_sc = 13
        for i in range(len(peak_width)):
            if abs(peak_width[i] - peak_width_mean) > 2:
                temp_sc -= 2
            elif abs(peak_width[i] - peak_width_mean) > 1:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[1] = temp_sc
        score += temp_sc
        # 第三個判斷 拍一下間隔的時間(差距盡量要相同 越連續) 共13分
        two_peak_maximum_pos_gap = np.array(two_peak_maximum_pos_gap)
        two_peak_maximum_pos_gap_mean = two_peak_maximum_pos_gap.mean()
        temp_sc = 13
        for i in range(len(two_peak_maximum_pos_gap)):
            if abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 3:
                temp_sc -= 2
            elif abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 2:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[2] = temp_sc
        score += temp_sc
        # 第四個判斷 拍一下的流暢度(拍下去 與回到初始位置的時間) 共13分
        st_to_max_to_end_diff = np.array(st_to_max_to_end_diff)
        temp_sc = 13
        for i in range(len(st_to_max_to_end_diff)):
            if abs(st_to_max_to_end_diff[i]) > 6:
                temp_sc -= 2
            elif abs(st_to_max_to_end_diff[i]) > 5:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[3] = temp_sc
        score += temp_sc
        print(f"score: {score}")
        if score >= 80:
            print(f"很棒")
        elif score >= 70:
            print(f"普通")
        else:
            print(f"很差")

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = LandmarkDataProcess.find_peak(raw_data, self.config)
        self.count_score(data)

class Action12:
    def __init__(self, path):
        self.config = {'pose': [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, raw_data):
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.config['pose']:
            for i in range(len(raw_data[point])):
                data.append(raw_data[point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position
        for i in range(len(data)):
            peak_width.append(data[i].end_pos - data[i].start_pos)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 12:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 5
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        temp_sc = 13
        for i in range(len(peak_width)):
            if abs(peak_width[i] - peak_width_mean) > 2:
                temp_sc -= 2
            elif abs(peak_width[i] - peak_width_mean) > 1:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[1] = temp_sc
        score += temp_sc
        # 第三個判斷 拍一下間隔的時間(差距盡量要相同 越連續) 共13分
        two_peak_maximum_pos_gap = np.array(two_peak_maximum_pos_gap)
        two_peak_maximum_pos_gap_mean = two_peak_maximum_pos_gap.mean()
        temp_sc = 13
        for i in range(len(two_peak_maximum_pos_gap)):
            if abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 3:
                temp_sc -= 2
            elif abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 2:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[2] = temp_sc
        score += temp_sc
        # 第四個判斷 拍一下的流暢度(拍下去 與回到初始位置的時間) 共13分
        st_to_max_to_end_diff = np.array(st_to_max_to_end_diff)
        temp_sc = 13
        for i in range(len(st_to_max_to_end_diff)):
            if abs(st_to_max_to_end_diff[i]) > 6:
                temp_sc -= 2
            elif abs(st_to_max_to_end_diff[i]) > 5:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[3] = temp_sc
        score += temp_sc
        print(f"score: {score}")
        if score >= 80:
            print(f"很棒")
        elif score >= 70:
            print(f"普通")
        else:
            print(f"很差")

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = LandmarkDataProcess.find_peak(raw_data, self.config)
        self.count_score(data)

class Action13:
    def __init__(self, path):
        self.config = {'pose': [11, 12, 13, 14, 15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, raw_data):
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.config['pose']:
            for i in range(len(raw_data[point])):
                data.append(raw_data[point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position
        for i in range(len(data)):
            peak_width.append(data[i].end_pos - data[i].start_pos)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 12:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 5
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        temp_sc = 13
        for i in range(len(peak_width)):
            if abs(peak_width[i] - peak_width_mean) > 2:
                temp_sc -= 2
            elif abs(peak_width[i] - peak_width_mean) > 1:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[1] = temp_sc
        score += temp_sc
        # 第三個判斷 拍一下間隔的時間(差距盡量要相同 越連續) 共13分
        two_peak_maximum_pos_gap = np.array(two_peak_maximum_pos_gap)
        two_peak_maximum_pos_gap_mean = two_peak_maximum_pos_gap.mean()
        temp_sc = 13
        for i in range(len(two_peak_maximum_pos_gap)):
            if abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 3:
                temp_sc -= 2
            elif abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 2:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[2] = temp_sc
        score += temp_sc
        # 第四個判斷 拍一下的流暢度(拍下去 與回到初始位置的時間) 共13分
        st_to_max_to_end_diff = np.array(st_to_max_to_end_diff)
        temp_sc = 13
        for i in range(len(st_to_max_to_end_diff)):
            if abs(st_to_max_to_end_diff[i]) > 6:
                temp_sc -= 2
            elif abs(st_to_max_to_end_diff[i]) > 5:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[3] = temp_sc
        score += temp_sc
        print(f"score: {score}")
        if score >= 80:
            print(f"很棒")
        elif score >= 70:
            print(f"普通")
        else:
            print(f"很差")

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = LandmarkDataProcess.find_peak(raw_data, self.config)
        self.count_score(data)

class Action14:
    def __init__(self, path):
        self.config = {'pose': [23, 24, 25, 26, 27, 28]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, raw_data):
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.config['pose']:
            for i in range(len(raw_data[point])):
                data.append(raw_data[point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position
        for i in range(len(data)):
            peak_width.append(data[i].end_pos - data[i].start_pos)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 12:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 5
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        temp_sc = 13
        for i in range(len(peak_width)):
            if abs(peak_width[i] - peak_width_mean) > 2:
                temp_sc -= 2
            elif abs(peak_width[i] - peak_width_mean) > 1:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[1] = temp_sc
        score += temp_sc
        # 第三個判斷 拍一下間隔的時間(差距盡量要相同 越連續) 共13分
        two_peak_maximum_pos_gap = np.array(two_peak_maximum_pos_gap)
        two_peak_maximum_pos_gap_mean = two_peak_maximum_pos_gap.mean()
        temp_sc = 13
        for i in range(len(two_peak_maximum_pos_gap)):
            if abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 3:
                temp_sc -= 2
            elif abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 2:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[2] = temp_sc
        score += temp_sc
        # 第四個判斷 拍一下的流暢度(拍下去 與回到初始位置的時間) 共13分
        st_to_max_to_end_diff = np.array(st_to_max_to_end_diff)
        temp_sc = 13
        for i in range(len(st_to_max_to_end_diff)):
            if abs(st_to_max_to_end_diff[i]) > 6:
                temp_sc -= 2
            elif abs(st_to_max_to_end_diff[i]) > 5:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[3] = temp_sc
        score += temp_sc
        print(f"score: {score}")
        if score >= 80:
            print(f"很棒")
        elif score >= 70:
            print(f"普通")
        else:
            print(f"很差")

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = LandmarkDataProcess.find_peak(raw_data, self.config)
        self.count_score(data)

class Action15:
    def __init__(self, path):
        self.config = {'pose': [23, 24, 25, 26, 27, 28]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def count_score(self, raw_data):
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.config['pose']:
            for i in range(len(raw_data[point])):
                data.append(raw_data[point][i])
        data.sort(key=lambda x: x.peak_max_pos)  # sort by peak maximum position
        for i in range(len(data)):
            peak_width.append(data[i].end_pos - data[i].start_pos)
            st_to_max_to_end_diff.append(
                abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        """輸出測試"""
        # print(f"peak_width {peak_width}")
        # print(f"two_peak_maximum_pos_gap {two_peak_maximum_pos_gap}")
        # print(f"two_peak_distance {two_peak_distance}")
        # print(f"st_to_max_to_end_diff {st_to_max_to_end_diff}")

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 12:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 5
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = np.array(peak_width)
        peak_width_mean = peak_width.mean()
        temp_sc = 13
        for i in range(len(peak_width)):
            if abs(peak_width[i] - peak_width_mean) > 2:
                temp_sc -= 2
            elif abs(peak_width[i] - peak_width_mean) > 1:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[1] = temp_sc
        score += temp_sc
        # 第三個判斷 拍一下間隔的時間(差距盡量要相同 越連續) 共13分
        two_peak_maximum_pos_gap = np.array(two_peak_maximum_pos_gap)
        two_peak_maximum_pos_gap_mean = two_peak_maximum_pos_gap.mean()
        temp_sc = 13
        for i in range(len(two_peak_maximum_pos_gap)):
            if abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 3:
                temp_sc -= 2
            elif abs(two_peak_maximum_pos_gap[i] - two_peak_maximum_pos_gap_mean) > 2:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[2] = temp_sc
        score += temp_sc
        # 第四個判斷 拍一下的流暢度(拍下去 與回到初始位置的時間) 共13分
        st_to_max_to_end_diff = np.array(st_to_max_to_end_diff)
        temp_sc = 13
        for i in range(len(st_to_max_to_end_diff)):
            if abs(st_to_max_to_end_diff[i]) > 6:
                temp_sc -= 2
            elif abs(st_to_max_to_end_diff[i]) > 5:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[3] = temp_sc
        score += temp_sc
        print(f"score: {score}")
        if score >= 80:
            print(f"很棒")
        elif score >= 70:
            print(f"普通")
        else:
            print(f"很差")

    def main_func(self):
        mdp = MDP()
        raw_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = LandmarkDataProcess.find_peak(raw_data, self.config)
        self.count_score(data)

#測試用
if __name__ == "__main__":
    target_action = 1
    target_dict = {
        1: Action1,
        2: Action2,
        3: Action3,
        4: Action4,
        5: Action5,
        6: Action6,
        7: Action7,
        8: Action8,
        9: Action9,
        10: Action10,
        11: Action11,
        12: Action12,
        13: Action13,
        14: Action14,
        15: Action15,
    }
    file_path = fr"C:\Bilateral Coordination Record Video\C001_2506021923\{target_action:02d}.mp4"
    if os.path.exists(file_path):
        target_dict[target_action](file_path).main_func()
    else:
        print(f"\033[91mPath Not Found.\033[0m")

    #C001_2506021854
    #C001_2506021923
    #C002_2506092042
