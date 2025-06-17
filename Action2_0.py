import os
import cv2
import numpy
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
import matplotlib.pyplot as plt


def pose_normalize(data):
    try:
        def dis(k1, k2):  # distance of two point
            d = pow(((k1[1] - k2[1]) * (k1[1] - k2[1]) + (k1[0] - k2[0]) * (k1[0] - k2[0])), .5)
            return d
        #normalization
        for frame in range(len(data)):
            ex_data = data[frame]['pose']
            unit = dis([ex_data[11]['x'], ex_data[11]['y']], [ex_data[23]['x'], ex_data[23]['y']])
            center = [(ex_data[11]['x'] + ex_data[12]['x']) / 2, (ex_data[11]['y'] + ex_data[12]['y']) / 2]
            for point in ex_data.keys():
                data[frame]['pose'][point]['x'] = (data[frame]['pose'][point]['x']-center[0])/unit
                data[frame]['pose'][point]['y'] = (data[frame]['pose'][point]['y']-center[1])/unit
    except Exception as ex:
        print(ex)
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
        print("Mediapipe Initialized")

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
                        num_hands=2)
                else:
                    options = config["option_class"](
                        base_options=self.base_options(model_asset_path=task_path),
                        running_mode=self.vision_running_mode.IMAGE)
                self.landmarkers[model_type] = config["landmarker_class"].create_from_options(options)
            except Exception as e:
                print(f"[Error] Failed to initialize {model_type}: {e}")

    def posepoint(self, x, y):
        return {"x": x, "y": y}

    def _process_video(self, video_path, use_models):
        cap = cv2.VideoCapture(video_path)
        data = {}
        count_image = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
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

                        data[count_image]["hand"] = {"Left": {}, "Right": {}}
                        for hand_label in ["Left", "Right"]:
                            if hand_label in detected_hands:
                                hand_landmarks = detected_hands[hand_label]
                                data[count_image]["hand"][hand_label] = {
                                    i: self.posepoint(lm.x, lm.y)
                                    for i, lm in enumerate(hand_landmarks)
                                }
                            else:
                                data[count_image]["hand"][hand_label] = {
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
        finally:
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
    def __init__(self, start=0.0, start_pos=0, end=0.0, end_pos=0, peak_max=0.0, peak_max_pos=0):
        self.start = start
        self.start_pos = start_pos
        self.end = end
        self.end_pos = end_pos
        self.peak_max = peak_max
        self.peak_max_pos = peak_max_pos

class Action1:
    def __init__(self, path):
        self.config = {'pose':[15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def find_peak(self, data, forward_find=0, mean_offset=0.01):
        info = {}  # return information
        for mt in self.config.keys():
            if mt == "pose":
                data = pose_normalize(data)
            # if mt == "face":
            #     data = face_normalize(data)
            # if mt == "hand":
            #     data = hands_normalize(data)
            for point in self.config[mt]:
                peak_detect = False
                temp = []
                process_data = [data[i][mt][point]['y'] for i in range(len(data))]
                process_data = numpy.array(process_data)
                data_mean = process_data.mean() + mean_offset  # 0.01 為了避免不相干的動作突破門檻
                print(f"data_mean : {data_mean}")
                peak_recorder = None
                for frame in range(len(data) - forward_find):
                    if not peak_detect and peak_recorder is None and data[frame + forward_find][mt][point]['y'] > data_mean:  # 偵測變化是否超過平均值
                        peak_detect = True
                        peak_recorder = PeakDataStruct(start=data[frame][mt][point]['y'], start_pos=frame)
                    elif peak_detect and peak_recorder is not None and data[frame][mt][point]['y'] < data_mean and frame > peak_recorder.start_pos + 5:
                        peak_detect = False
                        peak_recorder.end = data[frame + forward_find][mt][point]['y']
                        peak_recorder.end_pos = frame + forward_find
                        temp.append(peak_recorder)
                        peak_recorder = None # 重置 peak_recorder
                    if peak_detect:
                        # 確保只有在 peak_recorder 已正確初始化時才訪問它
                        try:
                            if data[frame][mt][point]['y'] > peak_recorder.peak_max:
                                peak_recorder.peak_max = data[frame][mt][point]['y']
                                peak_recorder.peak_max_pos = frame
                        except TypeError:
                            print("Here is no peak_recorder")

                info[point] = temp
        return info

    def count_score(self, raw_data):
        data = []
        peak_width = [] # 波的寬度
        two_peak_maximum_pos_gap = [] # 兩波峰的距離
        two_peak_distance = [] # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = [] #下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.config['pose']:
            for i in range(len(raw_data[point])):
                data.append(raw_data[point][i])
        data.sort(key = lambda x : x.peak_max_pos) # sort by peak maximum position
        for i in range(len(data)):
            peak_width.append(data[i].end_pos - data[i].start_pos)
            st_to_max_to_end_diff.append(abs((data[i].peak_max_pos - data[i].start_pos) - (data[i].end_pos - data[i].peak_max_pos)))
            if i != 0:
                two_peak_maximum_pos_gap.append(data[i].peak_max_pos - data[i - 1].peak_max_pos)
                two_peak_distance.append(data[i].start_pos - data[i - 1].end_pos)
        print(peak_width)
        print(two_peak_maximum_pos_gap)
        print(two_peak_distance)#目前沒使用
        print(st_to_max_to_end_diff)

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 6:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 10
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        #第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = numpy.array(peak_width)
        peak_width_mean = peak_width.mean()
        temp_sc = 13
        for i in range(len(peak_width)):
            if abs(peak_width[i] - peak_width_mean) > 2:
                temp_sc -= 2
            elif  abs(peak_width[i] - peak_width_mean) > 1:
                temp_sc -= 1
        if temp_sc < 0:
            temp_sc = 0
        self.score[1] = temp_sc
        score += temp_sc
        #第三個判斷 拍一下間隔的時間(差距盡量要相同 越連續) 共13分
        two_peak_maximum_pos_gap = numpy.array(two_peak_maximum_pos_gap)
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
        #第四個判斷 拍一下的流暢度(拍下去 與回到初始位置的時間) 共13分
        st_to_max_to_end_diff = numpy.array(st_to_max_to_end_diff)
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
        row_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = self.find_peak(row_data)
        self.count_score(data)

class Action2:
    def __init__(self, path):
        self.config = {'pose': [15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def find_peak(self, data, forward_find=0, mean_offset=0.01):
        info = {}  # return information
        for mt in self.config.keys():
            if mt == "pose":
                data = pose_normalize(data)
            # if mt == "face":
            #     data = face_normalize(data)
            # if mt == "hand":
            #     data = hands_normalize(data)
            for point in self.config[mt]:
                peak_detect = False
                temp = []
                process_data = [data[i][mt][point]['y'] for i in range(len(data))]
                process_data = numpy.array(process_data)
                data_mean = process_data.mean() + mean_offset  # 0.01 為了避免不相干的動作突破門檻
                print(f"data_mean : {data_mean}")
                peak_recorder = None
                for frame in range(len(data) - forward_find):
                    if not peak_detect and peak_recorder is None and data[frame + forward_find][mt][point][
                        'y'] > data_mean:  # 偵測變化是否超過平均值
                        peak_detect = True
                        peak_recorder = PeakDataStruct(start=data[frame][mt][point]['y'], start_pos=frame)
                    elif peak_detect and peak_recorder is not None and data[frame][mt][point][
                        'y'] < data_mean and frame > peak_recorder.start_pos + 5:
                        peak_detect = False
                        peak_recorder.end = data[frame + forward_find][mt][point]['y']
                        peak_recorder.end_pos = frame + forward_find
                        temp.append(peak_recorder)
                        peak_recorder = None  # 重置 peak_recorder
                    if peak_detect:
                        # 確保只有在 peak_recorder 已正確初始化時才訪問它
                        try:
                            if data[frame][mt][point]['y'] > peak_recorder.peak_max:
                                peak_recorder.peak_max = data[frame][mt][point]['y']
                                peak_recorder.peak_max_pos = frame
                        except TypeError:
                            print("Here is no peak_recorder")

                info[point] = temp
        return info

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
        print(peak_width)
        print(two_peak_maximum_pos_gap)
        print(two_peak_distance)  # 目前沒使用
        print(st_to_max_to_end_diff)

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 8:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 7.5
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = numpy.array(peak_width)
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
        two_peak_maximum_pos_gap = numpy.array(two_peak_maximum_pos_gap)
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
        st_to_max_to_end_diff = numpy.array(st_to_max_to_end_diff)
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
        row_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = self.find_peak(row_data)
        self.count_score(data)

class Action3:
    def __init__(self, path):
        self.config = {'pose': [15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def find_peak(self, data, forward_find=0, mean_offset=0.01):
        info = {}  # return information
        for mt in self.config.keys():
            if mt == "pose":
                data = pose_normalize(data)
            # if mt == "face":
            #     data = face_normalize(data)
            # if mt == "hand":
            #     data = hands_normalize(data)
            for point in self.config[mt]:
                peak_detect = False
                temp = []
                process_data = [data[i][mt][point]['y'] for i in range(len(data))]
                process_data = numpy.array(process_data)
                data_mean = process_data.mean() + mean_offset  # 0.01 為了避免不相干的動作突破門檻
                print(f"data_mean : {data_mean}")
                peak_recorder = None
                for frame in range(len(data) - forward_find):
                    if not peak_detect and peak_recorder is None and data[frame + forward_find][mt][point][
                        'y'] > data_mean:  # 偵測變化是否超過平均值
                        peak_detect = True
                        peak_recorder = PeakDataStruct(start=data[frame][mt][point]['y'], start_pos=frame)
                    elif peak_detect and peak_recorder is not None and data[frame][mt][point][
                        'y'] < data_mean and frame > peak_recorder.start_pos + 5:
                        peak_detect = False
                        peak_recorder.end = data[frame + forward_find][mt][point]['y']
                        peak_recorder.end_pos = frame + forward_find
                        temp.append(peak_recorder)
                        peak_recorder = None  # 重置 peak_recorder
                    if peak_detect:
                        # 確保只有在 peak_recorder 已正確初始化時才訪問它
                        try:
                            if data[frame][mt][point]['y'] > peak_recorder.peak_max:
                                peak_recorder.peak_max = data[frame][mt][point]['y']
                                peak_recorder.peak_max_pos = frame
                        except TypeError:
                            print("Here is no peak_recorder")

                info[point] = temp
        return info

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
        print(peak_width)
        print(two_peak_maximum_pos_gap)
        print(two_peak_distance)  # 目前沒使用
        print(st_to_max_to_end_diff)

        # score judgement
        # 第一個判斷 拍六下 每下10分 共60分
        temp_score = 60
        if len(data) == 6:
            score += temp_score
        else:
            temp_score -= abs(6 - len(data)) * 10
            score += max(temp_score, 0)
        self.score[0] = max(temp_score, 0)
        # 第二個判斷 每拍一下(上去+下來)的時間長度(差距越小越好) 共13分
        peak_width = numpy.array(peak_width)
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
        two_peak_maximum_pos_gap = numpy.array(two_peak_maximum_pos_gap)
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
        st_to_max_to_end_diff = numpy.array(st_to_max_to_end_diff)
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
        row_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = self.find_peak(row_data)
        self.count_score(data)

class Action4:
    def __init__(self, path):
        self.config = {'pose': [15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def find_peak(self, data, forward_find=0, mean_offset=0.01):
        info = {}  # return information
        for mt in self.config.keys():
            if mt == "pose":
                data = pose_normalize(data)
            # if mt == "face":
            #     data = face_normalize(data)
            # if mt == "hand":
            #     data = hands_normalize(data)
            for point in self.config[mt]:
                peak_detect = False
                temp = []
                process_data = [data[i][mt][point]['y'] for i in range(len(data))]
                process_data = numpy.array(process_data)
                data_mean = process_data.mean() + mean_offset  # 0.01 為了避免不相干的動作突破門檻
                print(f"data_mean : {data_mean}")
                peak_recorder = None
                for frame in range(len(data) - forward_find):
                    if not peak_detect and peak_recorder is None and data[frame + forward_find][mt][point][
                        'y'] > data_mean:  # 偵測變化是否超過平均值
                        peak_detect = True
                        peak_recorder = PeakDataStruct(start=data[frame][mt][point]['y'], start_pos=frame)
                    elif peak_detect and peak_recorder is not None and data[frame][mt][point][
                        'y'] < data_mean and frame > peak_recorder.start_pos + 5:
                        peak_detect = False
                        peak_recorder.end = data[frame + forward_find][mt][point]['y']
                        peak_recorder.end_pos = frame + forward_find
                        temp.append(peak_recorder)
                        peak_recorder = None  # 重置 peak_recorder
                    if peak_detect:
                        # 確保只有在 peak_recorder 已正確初始化時才訪問它
                        try:
                            if data[frame][mt][point]['y'] > peak_recorder.peak_max:
                                peak_recorder.peak_max = data[frame][mt][point]['y']
                                peak_recorder.peak_max_pos = frame
                        except TypeError:
                            print("Here is no peak_recorder")

                info[point] = temp
        return info

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
        print(peak_width)
        print(two_peak_maximum_pos_gap)
        print(two_peak_distance)  # 目前沒使用
        print(st_to_max_to_end_diff)

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
        peak_width = numpy.array(peak_width)
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
        two_peak_maximum_pos_gap = numpy.array(two_peak_maximum_pos_gap)
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
        st_to_max_to_end_diff = numpy.array(st_to_max_to_end_diff)
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
        row_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = self.find_peak(row_data)
        self.count_score(data)

class Action5:
    def __init__(self, path):
        self.config = {'pose': [15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(4)]

    def find_peak(self, data, forward_find=0, mean_offset=0.01):
        info = {}  # return information
        for mt in self.config.keys():
            if mt == "pose":
                data = pose_normalize(data)
            # if mt == "face":
            #     data = face_normalize(data)
            # if mt == "hand":
            #     data = hands_normalize(data)
            for point in self.config[mt]:
                peak_detect = False
                temp = []
                process_data = [data[i][mt][point]['y'] for i in range(len(data))]
                process_data = numpy.array(process_data)
                data_mean = process_data.mean() + mean_offset  # 0.01 為了避免不相干的動作突破門檻
                print(f"data_mean : {data_mean}")
                peak_recorder = None
                for frame in range(len(data) - forward_find):
                    if not peak_detect and peak_recorder is None and data[frame + forward_find][mt][point][
                        'y'] > data_mean:  # 偵測變化是否超過平均值
                        peak_detect = True
                        peak_recorder = PeakDataStruct(start=data[frame][mt][point]['y'], start_pos=frame)
                    elif peak_detect and peak_recorder is not None and data[frame][mt][point][
                        'y'] < data_mean and frame > peak_recorder.start_pos + 5:
                        peak_detect = False
                        peak_recorder.end = data[frame + forward_find][mt][point]['y']
                        peak_recorder.end_pos = frame + forward_find
                        temp.append(peak_recorder)
                        peak_recorder = None  # 重置 peak_recorder
                    if peak_detect:
                        # 確保只有在 peak_recorder 已正確初始化時才訪問它
                        try:
                            if data[frame][mt][point]['y'] > peak_recorder.peak_max:
                                peak_recorder.peak_max = data[frame][mt][point]['y']
                                peak_recorder.peak_max_pos = frame
                        except TypeError:
                            print("Here is no peak_recorder")

                info[point] = temp
        return info

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
        print(peak_width)
        print(two_peak_maximum_pos_gap)
        print(two_peak_distance)  # 目前沒使用
        print(st_to_max_to_end_diff)

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
        peak_width = numpy.array(peak_width)
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
        two_peak_maximum_pos_gap = numpy.array(two_peak_maximum_pos_gap)
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
        st_to_max_to_end_diff = numpy.array(st_to_max_to_end_diff)
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
        row_data = mdp.get_data(self.video_path, list(self.config.keys()))
        data = self.find_peak(row_data)
        self.count_score(data)


#測試用
if __name__ == "__main__":
    mdp  = MDP()
    landmark_config = {
        #"pose": [0, 11, 12],  # nose, left shoulder, right shoulder
        # "hand": [4, 8],  # thumb_tip, index_tip
        "face": [468, 473]  # left eye, chin
    }
    result = mdp.get_data(r"C:\Users\fangt\Downloads\01.mp4", landmark_config)
    print(result)
    mdp.close()