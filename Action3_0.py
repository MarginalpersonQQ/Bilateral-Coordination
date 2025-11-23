import os
from statistics import correlation
from turtledemo.penrose import start

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pylab as pl
from pygame.transform import threshold
from scipy.signal import find_peaks, peak_widths
from dtaidistance import dtw

# data_structure
# data[frame]["pose"or"face"]["pose"=>0~32or"face"=>0~467]['x'or'y']
# data[frame]["hand"]["left"or"right"][0~20]['x'or'y']
class MDP:
    def __init__(self):
        # 直接用 solutions.holistic
        self.mp_holistic = mp.solutions.holistic
        # 建立一個長存的 holistic 物件（可重複使用）
        # 參數可依需要微調：refine_face_landmarks=True 會多偵測 Iris，較耗時
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False
        )
        self.count_image = 0
        print("MediaPipe Holistic Initialized\n")

    @staticmethod
    def _interp_1d_list(vals):
        """對含 NaN 的 1D list 做線性插值；兩端用端點值延伸。"""
        arr = np.array(vals, dtype=float)
        idx = np.arange(arr.size)
        msk = ~np.isnan(arr)
        if msk.any():  # 全是 NaN 就跳過
            arr[~msk] = np.interp(idx[~msk], idx[msk], arr[msk])
        return arr.tolist()

    def interpolate_nans(self, data):
        """對 data[frame][type][point]['x'/'y'] 的 NaN 做線性插值。"""
        n_frames = len(data)

        # ---- Pose (33) ----
        if n_frames and "pose" in data[0]:
            for i in range(33):
                xs = [data[f]["pose"][i]["x"] for f in range(n_frames)]
                ys = [data[f]["pose"][i]["y"] for f in range(n_frames)]
                xs_i = self._interp_1d_list(xs)
                ys_i = self._interp_1d_list(ys)
                for f in range(n_frames):
                    data[f]["pose"][i]["x"] = xs_i[f]
                    data[f]["pose"][i]["y"] = ys_i[f]

        # ---- Hand (left/right, 21 each) ----
        if n_frames and "hand" in data[0]:
            for side in ("left", "right"):
                for i in range(21):
                    xs = [data[f]["hand"][side][i]["x"] for f in range(n_frames)]
                    ys = [data[f]["hand"][side][i]["y"] for f in range(n_frames)]
                    xs_i = self._interp_1d_list(xs)
                    ys_i = self._interp_1d_list(ys)
                    for f in range(n_frames):
                        data[f]["hand"][side][i]["x"] = xs_i[f]
                        data[f]["hand"][side][i]["y"] = ys_i[f]

        # ---- Face (468) ----
        if n_frames and "face" in data[0]:
            for i in range(468):
                xs = [data[f]["face"][i]["x"] for f in range(n_frames)]
                ys = [data[f]["face"][i]["y"] for f in range(n_frames)]
                xs_i = self._interp_1d_list(xs)
                ys_i = self._interp_1d_list(ys)
                for f in range(n_frames):
                    data[f]["face"][i]["x"] = xs_i[f]
                    data[f]["face"][i]["y"] = ys_i[f]
        return data

    @staticmethod
    def posepoint(x, y):
        return {"x": float(x), "y": float(y)}

    @staticmethod
    def _make_nan_points(n):
        # 生成 n 個 {'x':nan, 'y':nan}
        return {i: {"x": float("nan"), "y": float("nan")} for i in range(n)}

    def _copy_or_nan(self, data, frame_idx, key, n_points):
        # 沒有這個 key 的上一幀就補 NaN，有就複製上一幀
        if frame_idx > 0 and key in data.get(frame_idx - 1, {}):
            return data[frame_idx - 1][key].copy()
        return self._make_nan_points(n_points)

    def data_normalize(self, data):
        # try:
        def dis(k1, k2):  # distance of two point
            d = pow(((k1[1] - k2[1]) * (k1[1] - k2[1]) + (k1[0] - k2[0]) * (k1[0] - k2[0])), .5)
            return d

        # normalization
        for frame in range(len(data)):
            unit = dis([data[frame]['pose'][11]['x'], data[frame]['pose'][11]['y']],
                       [data[frame]['pose'][23]['x'], data[frame]['pose'][23]['y']])
            center = [(data[frame]['pose'][11]['x'] + data[frame]['pose'][12]['x']) / 2,
                      (data[frame]['pose'][11]['y'] + data[frame]['pose'][12]['y']) / 2]
            for type in data[frame].keys():
                if type == 'hand':
                    for hand_type in data[frame][type].keys():
                        for point in data[frame][type][hand_type].keys():
                            data[frame][type][hand_type][point]['x'] = (data[frame][type][hand_type][point]['x'] -
                                                                        center[0]) / unit
                            data[frame][type][hand_type][point]['y'] = (data[frame][type][hand_type][point]['y'] -
                                                                        center[1]) / unit
                else:
                    for point in data[frame][type].keys():
                        data[frame][type][point]['x'] = (data[frame][type][point]['x'] - center[0]) / unit
                        data[frame][type][point]['y'] = (data[frame][type][point]['y'] - center[1]) / unit
        # except Exception as ex:
        #     print(f"ERROR: data_normalize {ex}")
        return data

    def _process_video(self, video_path, use_models):
        if isinstance(video_path, str):
            cap = cv2.VideoCapture(video_path)
            use_cap = True
        else:
            frames = video_path
            use_cap = False

        data = {}
        count_image = 0

        try:
            while True:
                if use_cap:
                    ret, frame = cap.read()
                    if not ret:
                        print("Message: No Frame In Here.")
                        break
                else:
                    if count_image >= len(frames):
                        break
                    frame = frames[count_image]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 直接餵 numpy RGB
                results = self.holistic.process(rgb)

                data[count_image] = {}

                # ----- Pose (33) -----
                if "pose" in use_models:
                    if results.pose_landmarks and results.pose_landmarks.landmark:
                        lm = results.pose_landmarks.landmark
                        data[count_image]["pose"] = {
                            i: self.posepoint(p.x, p.y) for i, p in enumerate(lm)
                        }
                    else:
                        # 若該幀沒偵測到，沿用上一幀；如果沒有上一幀就補 NaN
                        data[count_image]["pose"] = self._copy_or_nan(
                            data, count_image, "pose", 33
                        )

                # ----- Hand (Left/Right, 21 each) -----
                if "hand" in use_models:
                    data[count_image]["hand"] = {"left": {}, "right": {}}

                    # Left
                    if results.left_hand_landmarks and results.left_hand_landmarks.landmark:
                        lm = results.left_hand_landmarks.landmark
                        data[count_image]["hand"]["left"] = {
                            i: self.posepoint(p.x, p.y) for i, p in enumerate(lm)
                        }
                    else:
                        # 手的資料一律補 NaN（跟你原本設計一致）
                        data[count_image]["hand"]["left"] = self._make_nan_points(21)

                    # Right
                    if results.right_hand_landmarks and results.right_hand_landmarks.landmark:
                        lm = results.right_hand_landmarks.landmark
                        data[count_image]["hand"]["right"] = {
                            i: self.posepoint(p.x, p.y) for i, p in enumerate(lm)
                        }
                    else:
                        data[count_image]["hand"]["right"] = self._make_nan_points(21)

                # ----- Face (468) -----
                if "face" in use_models:
                    if results.face_landmarks and results.face_landmarks.landmark:
                        lm = results.face_landmarks.landmark
                        data[count_image]["face"] = {
                            i: self.posepoint(p.x, p.y) for i, p in enumerate(lm)
                        }
                    else:
                        data[count_image]["face"] = self._copy_or_nan(
                            data, count_image, "face", 468
                        )

                count_image += 1

        except Exception as E:
            print(f"\033[93mexception: {E}\033[0m")
        finally:
            print(f"Mediapipe processed images : {count_image}")
            self.count_image = count_image
            if use_cap:
                cap.release()
        return data

    def show_test_image(self, data):
        pass

    def get_data(self, video_path, models=("pose", "hand", "face"), normalize = True):
        """
        models: e.g. ("pose",) or ("pose","hand")
        """
        raw_data = self._process_video(video_path, models)
        filled = self.interpolate_nans(raw_data)
        if normalize:
            norm_data = self.data_normalize(filled)
            return norm_data
        else:
            return filled

class MATCH_FUNC:
    #data_structur
    #space_pose{model:{point:[]}}(one axis)
    @staticmethod
    def space_position(norm_data, config, axis):
        space_pose = {}
        for model in config.keys():
            if model == "hand":
                space_pose[model] = {"left":{}, "right":{}}
            else:
                space_pose[model] = {}
            for point in config[model]:
                if model == "hand":
                    space_pose[model]["left"][point] = []
                    space_pose[model]["right"][point] = []
                else:
                    space_pose[model][point] = []
                for frame in range(len(norm_data)):
                    if model == "hand":
                        for hand in ['left', 'right']:
                            space_pose[model][hand][point].append(norm_data[frame][model][hand][point][axis])
                    else:
                        space_pose[model][point].append(norm_data[frame][model][point][axis])
        return space_pose

    #data_structur
    #time_sp{model:{point:[]}}(one axis)
    @staticmethod
    def time_speed(space_pos_data):
        time_sp = {}
        for model in space_pos_data.keys():
            if model != "hand":
                time_sp[model] = {}
                for point in space_pos_data[model].keys():
                    time_sp[model][point] = []
                    for frame in range(2, len(space_pos_data[model][point]), 2):
                        time_sp[model][point].append(space_pos_data[model][point][frame] - space_pos_data[model][point][frame-2])
            else:
                time_sp["hand"] = {"left" : {}, "right" : {}}
                for hand in space_pos_data[model].keys():
                    for point in space_pos_data[model][hand].keys():
                        time_sp[model][hand][point] = []
                        for frame in range(2, len(space_pos_data[model][hand][point]), 2):
                            time_sp[model][hand][point].append(space_pos_data[model][hand][point][frame] - space_pos_data[model][hand][point][frame-2])
        return time_sp

    @staticmethod
    def find_forward(raw_data, point_info):
        # Move forward to find a value close to 0
        forward_record = {}
        for model in point_info.keys():
            if model != "hand":
                forward_record[model] = {}
                for point in point_info[model].keys():
                    data_start_temp = None
                    for i in range(point_info[model][point][0][1], -1, -1):
                        if data_start_temp == None:
                            data_start_temp = [i, raw_data[model][point][i]]
                        elif data_start_temp[1] > abs(raw_data[model][point][i]):
                            data_start_temp = [i, raw_data[model][point][i]]
                        else:
                            break
                    forward_record[model][point] = data_start_temp
            else:
                forward_record[model] = {"left":{}, "right":{}}
                for hand in ['left', "right"]:
                    for point in point_info[model][hand].keys():
                        data_start_temp = None
                        for i in range(point_info[model][hand][point][0][1], -1, -1):
                            if data_start_temp == None:
                                data_start_temp = [i, raw_data[model][hand][point][i]]
                            elif data_start_temp[1] > abs(raw_data[model][hand][point][i]):
                                data_start_temp = [i, raw_data[model][hand][point][i]]
                            else:
                                break
                        forward_record[model][hand][point] = data_start_temp
        return forward_record

    @staticmethod
    def find_backward(raw_data, point_info):
        # Move backward to find a value close to 0
        backward_record = {}
        for model in point_info.keys():
            if model != "hand":
                backward_record[model] = {}
                for point in point_info[model].keys():
                    data_start_temp = None
                    for i in range(point_info[model][point][-1][1], len(raw_data[model][point])):
                        if data_start_temp == None:
                            data_start_temp = [i, raw_data[model][point][i]]
                        elif data_start_temp[1] > abs(raw_data[model][point][i]):
                            data_start_temp = [i, raw_data[model][point][i]]
                        else:
                            break
                    backward_record[model][point] = data_start_temp
            else:
                backward_record[model] = {"left": {}, "right": {}}
                for hand in ['left', "right"]:
                    for point in point_info[model][hand].keys():
                        data_start_temp = None
                        for i in range(point_info[model][hand][point][-1][1], len(raw_data[model][hand][point])):
                            if data_start_temp == None:
                                data_start_temp = [i, raw_data[model][hand][point][i]]
                            elif data_start_temp[1] > abs(raw_data[model][hand][point][i]):
                                data_start_temp = [i, raw_data[model][hand][point][i]]
                            else:
                                break
                        backward_record[model][hand][point] = data_start_temp
        return backward_record

class Action1:
    def __init__(self, path):
        self.config = {'pose':[15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(3)]

    def count_score(self, norm_data):
        #region feature_extraction
        space_y = MATCH_FUNC.space_position(norm_data, self.config, "y")
        time_y = MATCH_FUNC.time_speed(space_y)
        #endregion

        #region moving_side_judge
        movie_sequence = {} # score 1
        point_sequence = {}
        for model in time_y.keys():
            movie_sequence[model] = []
            point_sequence[model] = {}
            for point in time_y[model].keys():
                point_sequence[model][point] = []
                min_moving_speed = max(time_y[model][point]) * 0.25
                find = False
                temp_info = None
                for i, value in enumerate(time_y[model][point]):
                    if value >= min_moving_speed and find == False:
                        find = True
                        temp_info = [point, i, value]
                    elif value >= min_moving_speed and find == True:
                        if value > temp_info[2]:
                            temp_info = [point, i, value]
                    elif value < min_moving_speed and find == True:
                        find = False
                        movie_sequence[model].append(temp_info)
                        point_sequence[model][point].append(temp_info)
                        temp_info = None

            # movie_sequence[[pos, point, value]]
            movie_sequence[model].sort(key=lambda x:x[1])

        #region draw
        # plt.subplot(2, 1, 1)
        # plt.plot(time_y['pose'][15])
        # plt.axline((0, max(time_y['pose'][15]) * 0.25), (len(time_y['pose'][15]), max(time_y['pose'][15]) * 0.25))
        # plt.subplot(2, 1, 2)
        # plt.plot(time_y['pose'][16])
        # plt.axline((0, max(time_y['pose'][16]) * 0.25), (len(time_y['pose'][16]), max(time_y['pose'][16]) * 0.25))
        # plt.show()
        #endregion

        start_info = MATCH_FUNC.find_forward(time_y, point_sequence) # score 2 and 3
        finish_info = MATCH_FUNC.find_backward(time_y, point_sequence) # score 2 and 3
        #endregion

        #region score count
        #score1
        wrong = 0
        score_temp = []
        for model in movie_sequence.keys():
            for i in range(1, len(movie_sequence[model])):
                if movie_sequence[model][i-1][0] == movie_sequence[model][i][0]:
                    wrong += 1
            score_temp.append( (len(movie_sequence[model])-wrong) / len(movie_sequence[model]) * 100 )
        print(f"score1 {score_temp}")
        self.score[0] = np.array(score_temp).mean()
        #score2
        score_temp = []
        for model in space_y.keys():
            temp_array = []
            for point in space_y[model].keys():
                temp_array.append(space_y[model][point][start_info[model][point][0]*2 : finish_info[model][point][0]*2+3])
            min_len = min(map(len, temp_array))
            coef = np.corrcoef(temp_array[0][:min_len], temp_array[1][:min_len])[0, 1]
            score_temp.append(coef)
        print(f"score2 {score_temp}")
        self.score[1] = max(0, np.array(score_temp).mean() * 100)
        #score3
        score_temp = []
        for model in time_y.keys():
            temp_array = []
            for point in time_y[model].keys():
                temp_array.append(time_y[model][point][start_info[model][point][0]:finish_info[model][point][0]])
            min_len = min(map(len, temp_array))
            coef = np.corrcoef(temp_array[0][:min_len], temp_array[1][:min_len])[0, 1]
            score_temp.append(coef)
        print(f"score3 {score_temp}")
        self.score[2] = max(0, np.array(score_temp).mean() * 100)
        #endregion

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action2:
    def __init__(self, path):
        self.config = {'pose': [15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(3)]

    def count_score(self, norm_data):
        # region feature_extraction
        space_y = MATCH_FUNC.space_position(norm_data, self.config, "y")
        time_y = MATCH_FUNC.time_speed(space_y)
        # endregion

        # region moving_side_judge
        movie_sequence = {}  # score 1
        point_sequence = {}
        for model in time_y.keys():
            movie_sequence[model] = []
            point_sequence[model] = {}
            for point in time_y[model].keys():
                point_sequence[model][point] = []
                min_moving_speed = max(time_y[model][point]) * 0.25
                find = False
                temp_info = None
                for i, value in enumerate(time_y[model][point]):
                    if value >= min_moving_speed and find == False:
                        find = True
                        temp_info = [point, i, value]
                    elif value >= min_moving_speed and find == True:
                        if value > temp_info[2]:
                            temp_info = [point, i, value]
                    elif value < min_moving_speed and find == True:
                        find = False
                        movie_sequence[model].append(temp_info)
                        point_sequence[model][point].append(temp_info)
                        temp_info = None

            # movie_sequence[[pos, point, value]]
            movie_sequence[model].sort(key=lambda x: x[1])

        #region draw
        # plt.subplot(2, 1, 1)
        # plt.plot(time_y['pose'][15])
        # plt.axline((0, max(time_y['pose'][15]) * 0.25), (len(time_y['pose'][15]), max(time_y['pose'][15]) * 0.25))
        # plt.subplot(2, 1, 2)
        # plt.plot(time_y['pose'][16])
        # plt.axline((0, max(time_y['pose'][16]) * 0.25), (len(time_y['pose'][16]), max(time_y['pose'][16]) * 0.25))
        # plt.show()
        #endregion

        start_info = MATCH_FUNC.find_forward(time_y, point_sequence)  # score 2 and 3
        finish_info = MATCH_FUNC.find_backward(time_y, point_sequence)  # score 2 and 3
        # endregion

        # region score count
        # score1
        wrong = 0
        score_temp = []
        for model in movie_sequence.keys():
            previous_site = None
            previous_site_length = 0
            # set_examine
            for i in movie_sequence[model]:
                if previous_site is None or previous_site != i[0]:
                    if previous_site is not None and previous_site_length < 2:
                        wrong += 1
                    previous_site = i[0]
                    previous_site_length = 1
                elif previous_site == i[0]:
                    previous_site_length += 1
                    if previous_site_length > 2:
                        print(i)
                        wrong += 1
            score_temp.append((len(movie_sequence[model])  - wrong) / len(movie_sequence[model]) * 100)
        print(f"score1 {score_temp}")
        self.score[0] = np.array(score_temp).mean()
        # score2
        score_temp = []
        for model in space_y.keys():
            temp_array = []
            for point in space_y[model].keys():
                temp_array.append(
                    space_y[model][point][start_info[model][point][0] * 2: finish_info[model][point][0] * 2 + 3])
            min_len = min(map(len, temp_array))
            coef = np.corrcoef(temp_array[0][:min_len], temp_array[1][:min_len])[0, 1]
            score_temp.append(coef)
        print(f"score2 {score_temp}")
        self.score[1] = max(0, np.array(score_temp).mean() * 100)
        # score3
        score_temp = []
        for model in time_y.keys():
            temp_array = []
            for point in time_y[model].keys():
                temp_array.append(time_y[model][point][start_info[model][point][0]:finish_info[model][point][0]])
            min_len = min(map(len, temp_array))
            coef = np.corrcoef(temp_array[0][:min_len], temp_array[1][:min_len])[0, 1]
            score_temp.append(coef)
        print(f"score3 {score_temp}")
        self.score[2] = max(0, np.array(score_temp).mean() * 100)
        # endregion

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action3:
    def __init__(self, path):
        self.config = {'pose': [15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(1)]

    def count_score(self, norm_data):
        # region feature_extraction
        space_y = MATCH_FUNC.space_position(norm_data, self.config, "y")
        time_y = MATCH_FUNC.time_speed(space_y)
        # endregion

        # region moving_side_judge
        movie_sequence = {}  # score 1
        point_sequence = {}
        for model in time_y.keys():
            movie_sequence[model] = []
            point_sequence[model] = {}
            for point in time_y[model].keys():
                point_sequence[model][point] = []
                min_moving_speed = max(time_y[model][point]) * 0.25
                find = False
                temp_info = None
                for i, value in enumerate(time_y[model][point]):
                    if value >= min_moving_speed and find == False:
                        find = True
                        temp_info = [point, i, value]
                    elif value >= min_moving_speed and find == True:
                        if value > temp_info[2]:
                            temp_info = [point, i, value]
                    elif value < min_moving_speed and find == True:
                        find = False
                        movie_sequence[model].append(temp_info)
                        point_sequence[model][point].append(temp_info)
                        temp_info = None

            # movie_sequence[[pos, point, value]]
            movie_sequence[model].sort(key=lambda x: x[1])
        start_info = MATCH_FUNC.find_forward(time_y, point_sequence)  # score 2 and 3
        finish_info = MATCH_FUNC.find_backward(time_y, point_sequence)  # score 2 and 3
        # region draw
        # plt.subplot(2, 1, 1)
        # plt.plot(time_y['pose'][15])
        # plt.axline((0, max(time_y['pose'][15]) * 0.25), (len(time_y['pose'][15]), max(time_y['pose'][15]) * 0.25))
        # plt.axline((start_info['pose'][15][0], 0.1), (start_info['pose'][15][0], -0.1))
        # plt.axline((finish_info['pose'][15][0], 0.1), (finish_info['pose'][15][0], -0.1))
        # plt.subplot(2, 1, 2)
        # plt.plot(time_y['pose'][16])
        # plt.axline((0, max(time_y['pose'][16]) * 0.25), (len(time_y['pose'][16]), max(time_y['pose'][16]) * 0.25))
        # plt.axline((start_info['pose'][16][0], 0.1), (start_info['pose'][16][0], -0.1))
        # plt.axline((finish_info['pose'][16][0], 0.1), (finish_info['pose'][16][0], -0.1))
        # plt.show()
        # endregion
        # endregion

        # region score count
        # score1
        score_temp = []
        sequences = [[15, 16, 16, 16, 15, 15], [16, 15, 15, 15, 16, 16]]
        for pos in range(5, len(movie_sequence["pose"])):
            for sequence in sequences:
                count = 0
                temp_mv = []
                for i in range(pos-5,pos+1):
                    temp_mv.append(movie_sequence['pose'][i][0])
                for seq in zip(sequence, temp_mv):
                    if seq[0] == seq[1]:
                        count += 1
                score_temp.append(count)
        print(f"score1 {max(score_temp) / 6 * 100}")
        self.score[0] = max(score_temp) / 6 * 100
        # endregion

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action4:
    def __init__(self, path):
        self.config = {'pose': [15, 16], 'hand' : [4, 20]}
        self.video_path = path
        self.score = [0 for _ in range(2)]

    def count_score(self, norm_data):
        # region feature_extraction
        space_y = MATCH_FUNC.space_position(norm_data, self.config, "y")
        space_x = MATCH_FUNC.space_position(norm_data, self.config, "x")
        time_y = MATCH_FUNC.time_speed(space_y)
        time_x = MATCH_FUNC.time_speed(space_x)
        # endregion

        # region moving_side_judge
        # region axis y
        movie_sequence_y = {}  # score 1
        point_sequence_y = {}
        for model in time_y.keys():
            if model != "hand":
                movie_sequence_y[model] = []
                point_sequence_y[model] = {}
                for point in time_y[model].keys():
                    point_sequence_y[model][point] = []
                    min_moving_speed = max(time_y[model][point]) * 0.25
                    find = False
                    temp_info = None
                    for i, value in enumerate(time_y[model][point]):
                        if value >= min_moving_speed and find == False:
                            find = True
                            temp_info = [point, i, value]
                        elif value >= min_moving_speed and find == True:
                            if value > temp_info[2]:
                                temp_info = [point, i, value]
                        elif value < min_moving_speed and find == True:
                            find = False
                            movie_sequence_y[model].append(temp_info)
                            point_sequence_y[model][point].append(temp_info)
                            temp_info = None
                # movie_sequence[[pos, point, value]]
                movie_sequence_y[model].sort(key=lambda x: x[1])
            else:
                movie_sequence_y[model] = []
                point_sequence_y[model] = {"left":{}, "right":{}}
                for hand in ["left", "right"]:
                    for point in time_y[model][hand].keys():
                        point_sequence_y[model][hand][point] = []
                        min_moving_speed = max(time_y[model][hand][point]) * 0.25
                        find = False
                        temp_info = None
                        for i, value in enumerate(time_y[model][hand][point]):
                            if value >= min_moving_speed and find == False:
                                find = True
                                temp_info = [point, i, value, hand]
                            elif value >= min_moving_speed and find == True:
                                if value > temp_info[2]:
                                    temp_info = [point, i, value, hand]
                            elif value < min_moving_speed and find == True:
                                find = False
                                movie_sequence_y[model].append(temp_info)
                                point_sequence_y[model][hand][point].append(temp_info)
                                temp_info = None
                    # movie_sequence[[pos, point, value, hand]]
                    movie_sequence_y[model].sort(key=lambda x: x[1])
        #endregion
        # region axis x
        movie_sequence_x = {}  # score 1
        point_sequence_x = {}
        for model in time_x.keys():
            if model != "hand":
                movie_sequence_x[model] = []
                point_sequence_x[model] = {}
                for point in time_y[model].keys():
                    point_sequence_x[model][point] = []
                    min_moving_speed = max(time_x[model][point]) * 0.25
                    find = False
                    temp_info = None
                    for i, value in enumerate(time_y[model][point]):
                        if value >= min_moving_speed and find == False:
                            find = True
                            temp_info = [point, i, value]
                        elif value >= min_moving_speed and find == True:
                            if value > temp_info[2]:
                                temp_info = [point, i, value]
                        elif value < min_moving_speed and find == True:
                            find = False
                            movie_sequence_x[model].append(temp_info)
                            point_sequence_x[model][point].append(temp_info)
                            temp_info = None
                # movie_sequence[[pos, point, value]]
                movie_sequence_x[model].sort(key=lambda x: x[1])
            else:
                movie_sequence_x[model] = []
                point_sequence_x[model] = {"left": {}, "right": {}}
                for hand in ["left", "right"]:
                    for point in time_y[model][hand].keys():
                        point_sequence_x[model][hand][point] = []
                        min_moving_speed = max(time_y[model][hand][point]) * 0.25
                        find = False
                        temp_info = None
                        for i, value in enumerate(time_y[model][hand][point]):
                            if value >= min_moving_speed and find == False:
                                find = True
                                temp_info = [point, i, value, hand]
                            elif value >= min_moving_speed and find == True:
                                if value > temp_info[2]:
                                    temp_info = [point, i, value, hand]
                            elif value < min_moving_speed and find == True:
                                find = False
                                movie_sequence_x[model].append(temp_info)
                                point_sequence_x[model][hand][point].append(temp_info)
                                temp_info = None
                    # movie_sequence[[pos, point, value, hand]]
                    movie_sequence_x[model].sort(key=lambda x: x[1])
        #endregion
        #region draw
        # plt.subplot(3, 2, 1)
        # plt.plot(time_y['pose'][15])
        # plt.axline((0, max(time_y['pose'][15]) * 0.25), (len(time_y['pose'][15]), max(time_y['pose'][15]) * 0.25))
        # plt.subplot(3, 2, 2)
        # plt.plot(time_y['pose'][16])
        # plt.axline((0, max(time_y['pose'][16][:-20]) * 0.25), (len(time_y['pose'][16][:-20]), max(time_y['pose'][16][:-20]) * 0.25))
        # plt.subplot(3, 2, 3)
        # plt.plot(time_x['hand']['left'][4][20:-20])
        # plt.axline((0, max(time_x['hand']['left'][4][:-20]) * 0.25),(len(time_x['hand']['left'][4]), max(time_x['hand']['left'][4][:-20]) * 0.25))
        # plt.subplot(3, 2, 4)
        # plt.plot(time_x['hand']['left'][20][20:-20])
        # plt.axline((0, max(time_x['hand']['left'][20][:-20]) * 0.25),(len(time_x['hand']['left'][20]), max(time_x['hand']['left'][20][:-20]) * 0.25))
        # plt.subplot(3, 2, 5)
        # plt.plot(time_x['hand']['right'][4][20:-20])
        # plt.axline((0, max(time_x['hand']['right'][4][:-20]) * 0.25),(len(time_x['hand']['right'][4]), max(time_x['hand']['right'][4][:-20]) * 0.25))
        # plt.subplot(3, 2, 6)
        # plt.plot(time_x['hand']['right'][20][20:-20])
        # plt.axline((0, max(time_x['hand']['right'][20][:-20]) * 0.25),(len(time_x['hand']['right'][20]), max(time_x['hand']['right'][20][:-20]) * 0.25))
        # plt.show()
        #endregion

        start_info_y = MATCH_FUNC.find_forward(time_y, point_sequence_y)  # score 2 and 3
        finish_info_y = MATCH_FUNC.find_backward(time_y, point_sequence_y)  # score 2 and 3
        start_info_x = MATCH_FUNC.find_forward(time_y, point_sequence_y)  # score 2 and 3
        finish_info_x = MATCH_FUNC.find_backward(time_y, point_sequence_y)  # score 2 and 3
        # endregion

        # region score count
        # score1
        min_len = min(len(time_y['pose'][16]), len(time_y['pose'][15]))
        coef_time = np.corrcoef(time_y['pose'][16][:min_len], time_y['pose'][15][:min_len])[0, 1]
        min_len = min(len(space_y['pose'][16]), len(space_y['pose'][15]))
        coef_space = np.corrcoef(space_y['pose'][16][:min_len], space_y['pose'][15][:min_len])[0, 1]
        print(f"score1 {(coef_time * 100 + coef_space * 100) / 2}")
        self.score[0] = max(0, (coef_time * 100 + coef_space * 100) / 2)

        # score2
        # min_len = min(len(space_x['hand']['left'][4]), len(space_x['hand']['right'][4]))
        # coef = np.corrcoef(space_x['hand']['left'][4][:min_len], space_x['hand']['right'][4][:min_len])[0, 1]
        # left_score = abs(coef * 100)
        # min_len = min(len(space_x['hand']['left'][20]), len(space_x['hand']['right'][20]))
        # coef = np.corrcoef(space_x['hand']['left'][20][:min_len], space_x['hand']['right'][20][:min_len])[0, 1]
        # right_score = abs(coef * 100)
        #
        # min_len = min(len(time_x['hand']['left'][4]), len(time_x['hand']['right'][4]))
        # coef = np.corrcoef(time_x['hand']['left'][4][:min_len], time_x['hand']['right'][4][:min_len])[0, 1]
        # left_score = max(abs(coef * 100), left_score)
        # min_len = min(len(time_x['hand']['left'][20]), len(time_x['hand']['right'][20]))
        # coef = np.corrcoef(time_x['hand']['left'][20][:min_len], time_x['hand']['right'][20][:min_len])[0, 1]
        # right_score = max(abs(coef * 100), right_score)
        # print(f"score2 {(left_score + right_score) / 2}")
        # self.score[1] = (left_score + right_score) / 2
        left_ = np.array(space_x['hand']['left'][4]) - np.array(space_x['hand']['left'][20])
        right_ = np.array(space_x['hand']['right'][4]) - np.array(space_x['hand']['right'][20])
        min_len = min(len(left_), len(right_))
        coef = np.corrcoef(left_[:min_len],right_[:min_len])[0, 1]
        print(f"score2 {abs(coef) * 100}")
        self.score[1] = abs(coef) * 100
        # endregion

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action5:
    def __init__(self, path):
        self.config = {'pose': [11, 12, 15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(2)]

    def count_score(self, norm_data):
        # region feature_extraction
        space_x = MATCH_FUNC.space_position(norm_data, self.config, "x")
        # space_y = MATCH_FUNC.space_position(norm_data, self.config, "y")
        # endregion
        dist_15_16 = (np.array(space_x["pose"][15]) - np.array(space_x["pose"][16]))
        dist_16_15 = (np.array(space_x["pose"][16]) - np.array(space_x["pose"][15]))
        dist_12_15 = np.array(space_x["pose"][12]) - np.array(space_x["pose"][15])
        dist_11_16 = np.array(space_x["pose"][11]) - np.array(space_x["pose"][16])

        # region draw
        # plt.subplot(3, 1, 1)
        # plt.plot(dist_15_16, label = "15-16", color = "green")
        # plt.plot(dist_16_15, label = "16-15", color = "red")
        # plt.axline((0, dist_15_16.mean()), (len(dist_15_16), dist_15_16.mean()), label="15_16_mean", color = "green")
        # plt.legend(loc="lower right")
        # plt.subplot(3, 1, 2)
        # plt.plot(dist_12_15, label = "12-15", color = "blue")
        # plt.axline((0, dist_12_15.mean()), (len(dist_12_15), dist_12_15.mean()), label="12_15_mean", color = "blue")
        # plt.legend(loc="lower right")
        # plt.subplot(3, 1, 3)
        # plt.plot(dist_11_16, label = "11-16", color = "red")
        # plt.axline((0, dist_11_16.mean()), (len(dist_11_16), dist_11_16.mean()), label="11_16_mean", color = "red")
        # plt.legend(loc="lower right")
        # plt.show()
        # endregion

        # region count score
        #score 1
        coef_1 = np.corrcoef(dist_11_16, dist_15_16)[0, 1]
        coef_2 = np.corrcoef(dist_12_15, dist_16_15)[0, 1]
        print(f"score 1 {((coef_1 + coef_2) / 2) * 100}")
        self.score[0] = max(0, ((coef_1 + coef_2) / 2) * 100)
        #score 2
        threshold_12_15 = min(np.abs(dist_12_15)) + 0.05
        threshold_11_16 = min(np.abs(dist_11_16)) + 0.05
        count_area_12_15 = 0
        count_area_11_16 = 0

        find_small = False
        abs_12_15 = np.abs(dist_12_15)
        for i in range(len(dist_12_15)):
            if abs_12_15[i] <= threshold_12_15 and find_small is False :
                find_small = True
                count_area_12_15 += 1
            elif abs_12_15[i] > threshold_12_15 and find_small is True:
                find_small = False
        abs_11_16 = np.abs(dist_11_16)
        for i in range(len(dist_11_16)):
            if abs_11_16[i] <= threshold_11_16 and find_small is False :
                find_small = True
                count_area_11_16 += 1
            elif abs_11_16[i] > threshold_11_16 and find_small is True:
                find_small = False
        score_temp = []
        if count_area_12_15 != 1:
            score_temp.append(0)
        else:
            score_temp.append(100)
        if count_area_11_16 != 1:
            score_temp.append(0)
        else:
            score_temp.append(100)
        print(f"score 2 {np.array(score_temp).mean()}")
        self.score[0] = max(0, np.array(score_temp).mean())
        # endregion

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action6:
    def __init__(self, path):
        self.config = {'pose': [15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(2)]

    def count_score(self, norm_data):
        #region feature_extraction
        space_x = MATCH_FUNC.space_position(norm_data, self.config, "x")
        time_x = MATCH_FUNC.time_speed(space_x)
        #endregion

        #region count score
        #score1
        coef = np.corrcoef(space_x["pose"][16], space_x["pose"][15])[0, 1]
        print(f"score1 {abs(coef * 100)}")
        self.score[0] = abs(coef * 100)
        #score2
        coef = np.corrcoef(time_x["pose"][16], time_x["pose"][15])[0, 1]
        print(f"score2 {abs(coef * 100)}")
        self.score[1]= abs(coef * 100)
        #end region

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action7:
    def __init__(self, path):
        self.config = {'pose': [11, 12, 13, 14, 15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(2)]

    def count_score(self, norm_data):
        #region feature_extraction
        space_x = MATCH_FUNC.space_position(norm_data, self.config, "x")
        time_x = MATCH_FUNC.time_speed(space_x)
        #endregion

        #region count score
        #score1
        coef = np.corrcoef(space_x["pose"][16], space_x["pose"][15])[0, 1]
        print(f"score1 {abs(coef * 100)}")
        self.score[0] = abs(coef * 100)
        #score2
        coef = np.corrcoef(time_x["pose"][16], time_x["pose"][15])[0, 1]
        print(f"score2 {abs(coef * 100)}")
        self.score[1]= abs(coef * 100)
        #end region

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action8:
    def __init__(self, path):
        self.config = {'pose': [11, 12, 13, 14, 15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(2)]

    def count_score(self, norm_data):
        #region feature_extraction
        space_x = MATCH_FUNC.space_position(norm_data, self.config, "x")
        time_x = MATCH_FUNC.time_speed(space_x)
        #endregion

        #region count score
        #score1
        coef = np.corrcoef(space_x["pose"][16], space_x["pose"][15])[0, 1]
        print(f"score1 {abs(coef * 100)}")
        self.score[0] = abs(coef * 100)
        #score2
        coef = np.corrcoef(time_x["pose"][16], time_x["pose"][15])[0, 1]
        print(f"score2 {abs(coef * 100)}")
        self.score[1]= abs(coef * 100)
        #end region

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action9:
    def __init__(self, path):
        self.config = {'hand' : [4, 8, 12, 16, 20]}
        self.video_path = path
        self.score = [0 for _ in range(2)]

    def count_score(self, data, amount_of_images):
        """
        thumb = 4
        index_figer = 8
        middle_figer = 12
        ring_figer = 16
        pinky_finger = 20
        """
        # [left 8 12 16 20 right 8 12 16 20]
        def two_finger_dis(finger_data, which_hand):
            finger_landmarks_list = [8, 12, 16, 20]
            thumb = 4
            dis_track = [[], [], [], []]
            for frame in range(amount_of_images):
                fig = finger_data[frame]["hand"][which_hand]
                for c, i in enumerate(finger_landmarks_list):
                    dis = ((fig[i]["x"]-fig[thumb]["x"]) ** 2 + (fig[i]["y"]-fig[thumb]["y"]) ** 2) ** 0.5
                    dis_track[c].append(dis)
            return dis_track

        def moving_avg(data, window_size=5):
            result = []
            for d in data:
                temp = []
                for i in range(window_size, len(d)+1):
                    temp.append(np.array(d[i-5:i]).mean())
                result.append(temp)
            return result

        def find_touch(avg_data):
            touch_map = []
            touch_time = []
            no_touch = True
            right_now_touch = None
            start_touch_time = 0

            for i in range(len(avg_data[0])):
                # 檢查是否有碰觸
                flag = False
                for j in range(4):
                    if avg_data[j][i] <= 0.015:
                        no_touch = False
                        flag = True
                        break
                # 如果從有碰觸變沒碰觸
                if not flag and not no_touch:
                    if start_touch_time != 0:
                        touch_time.append(i - start_touch_time // 2)
                        start_touch_time = 0
                    no_touch = True
                    right_now_touch = None

                # 有碰觸 找最近 並記錄
                if not no_touch:
                    temp_min = avg_data[0][i]
                    temp_touch_fig = 0
                    for j in range(1, 4):
                        if avg_data[j][i] < temp_min:
                            temp_min = avg_data[j][i]
                            temp_touch_fig = j
                    if right_now_touch is None or right_now_touch != temp_touch_fig:
                        right_now_touch = temp_touch_fig
                        touch_map.append(right_now_touch)
                        if start_touch_time != 0:
                            touch_time.append(i - start_touch_time // 2)
                        if right_now_touch != temp_touch_fig:
                            start_touch_time = 0
                    start_touch_time += 1
            return {"t_map":touch_map, "t_time":touch_time}

        def sliding_corr_max(a, b):
            """
            在兩個長度不同的序列 a, b 之間，
            嘗試所有滑動位置，找出最高的 Pearson correlation。

            回傳：
              best_corr : 最高相關
              best_i    : 對應的起點 index（短序列在長序列上的位置）
            """
            a = np.array(a, dtype=float)
            b = np.array(b, dtype=float)

            # 確保 a 是較長，b 是較短
            if len(a) < len(b):
                a, b = b, a  # swap

            lenA = len(a)
            lenB = len(b)

            best_corr = -1.0
            best_i = 0

            # 滑動 b 在 a 上
            for i in range(lenA - lenB + 1):
                window = a[i:i + lenB]

                if np.std(window) == 0 or np.std(b) == 0:
                    corr = 0
                else:
                    corr = np.corrcoef(window, b)[0, 1]

                if corr > best_corr:
                    best_corr = corr
                    best_i = i

            return best_corr, best_i

        #region check touch every fingers
        #right
        right_dis = two_finger_dis(data, "right")
        #left
        left_dis = two_finger_dis(data, "left")
        #endregion
        # sliding windows moving average
        right_avg = moving_avg(right_dis)
        left_avg = moving_avg(left_dis)
        # plt.subplot(2,1,2)
        # plt.plot(right_avg[0])
        # plt.plot(right_avg[1])
        # plt.plot(right_avg[2])
        # plt.plot(right_avg[3])
        # plt.show()
        #count score
        right_info = find_touch(right_avg)
        left_info = find_touch(left_avg)

        sequence = [0, 1, 2, 3, 3, 2, 1, 0]
        score_temp = []

        temp_mv = []
        max_count = 0
        print(right_info, left_info)
        #right
        for i in range(8, len(right_info["t_map"])+1):
            temp_mv = right_info["t_map"][i-8:i]
            count = 0
            for seq in zip(sequence, temp_mv):
                if seq[0] == seq[1]:
                    count += 1
            if count > max_count:
                max_count = count
        #left
        for i in range(8, len(left_info["t_map"]) + 1):
            temp_mv = left_info["t_map"][i - 8:i]
            count = 0
            for seq in zip(sequence, temp_mv):
                if seq[0] == seq[1]:
                    count += 1
            if count > max_count:
                max_count = count
        score_temp.append(max_count)
        self.score[0] = np.array(score_temp).mean() * 100/8
        print(f"score0: {self.score[0]}")
        corr, idx = sliding_corr_max(right_info["t_time"], left_info["t_time"])
        self.score[1] = abs(corr * 100)
        print(f"score0: {self.score[1]}")

    def hand_area_extract(self, pose_lm_data, video_path):
        # right hand landmarks number = [16, 18, 20, 22]
        # left hand landmarks number = [15, 17, 19, 21]
        # video resolution 1280 * 720
        # Select a 150x150 pixel area
        cap = cv2.VideoCapture(video_path)
        h = 1280
        w = 720
        cut_pixel = 120 // 2
        merged_frame = []
        for i in range(len(pose_lm_data)):
            ret, frame = cap.read()
            #right hand
            right_center_x = int((pose_lm_data[i]["pose"][16]["x"]+pose_lm_data[i]["pose"][18]["x"]+
                        pose_lm_data[i]["pose"][20]["x"]+pose_lm_data[i]["pose"][22]["x"]) / 4 * h)
            right_center_y = int((pose_lm_data[i]["pose"][16]["y"] + pose_lm_data[i]["pose"][18]["y"] +
                        pose_lm_data[i]["pose"][20]["y"] + pose_lm_data[i]["pose"][22]["y"]) / 4 * w)
            right_newframe = frame[right_center_y-cut_pixel:right_center_y+cut_pixel, right_center_x-cut_pixel:right_center_x+cut_pixel]
            #left hand
            left_center_x = int((pose_lm_data[i]["pose"][15]["x"]+pose_lm_data[i]["pose"][17]["x"]+
                        pose_lm_data[i]["pose"][19]["x"]+pose_lm_data[i]["pose"][21]["x"]) / 4 * h)
            left_center_y = int((pose_lm_data[i]["pose"][15]["y"] + pose_lm_data[i]["pose"][17]["y"] +
                        pose_lm_data[i]["pose"][19]["y"] + pose_lm_data[i]["pose"][21]["y"]) / 4 * w)
            left_newframe = frame[left_center_y-cut_pixel:left_center_y+cut_pixel, left_center_x-50:left_center_x+cut_pixel]
            #merged two hands frame
            merged_frame.append(np.concatenate((right_newframe, left_newframe), axis=1))
        cap.release()
        return merged_frame

    def main_func(self):
        mdp = MDP()
        # pose_data = mdp.get_data(self.video_path, "pose", normalize=False)
        # new_video = self.hand_area_extract(pose_data, self.video_path)
        hand_landmarks = mdp.get_data(self.video_path, "hand", normalize=False)
        self.count_score(hand_landmarks, mdp.count_image)

class Action10:
    def __init__(self, path):
        self.config = {'pose': [27, 28]}
        self.video_path = path
        self.score = [0 for _ in range(3)]

    def count_score(self, norm_data):
        # region feature_extraction
        space_y = MATCH_FUNC.space_position(norm_data, self.config, "y")
        time_y = MATCH_FUNC.time_speed(space_y)
        # endregion

        # region moving_side_judge
        movie_sequence = {}  # score 1
        point_sequence = {}
        for model in time_y.keys():
            movie_sequence[model] = []
            point_sequence[model] = {}
            for point in time_y[model].keys():
                point_sequence[model][point] = []
                min_moving_speed = max(time_y[model][point]) * 0.25
                find = False
                temp_info = None
                for i, value in enumerate(time_y[model][point]):
                    if value >= min_moving_speed and find == False:
                        find = True
                        temp_info = [point, i, value]
                    elif value >= min_moving_speed and find == True:
                        if value > temp_info[2]:
                            temp_info = [point, i, value]
                    elif value < min_moving_speed and find == True:
                        find = False
                        movie_sequence[model].append(temp_info)
                        point_sequence[model][point].append(temp_info)
                        temp_info = None

            # movie_sequence[[pos, point, value]]
            movie_sequence[model].sort(key=lambda x: x[1])

        # region draw
        # plt.subplot(2, 1, 1)
        # plt.plot(space_y['pose'][27])
        # plt.axline((0, max(space_y['pose'][27]) * 0.25), (len(space_y['pose'][27]), max(space_y['pose'][27]) * 0.25))
        # plt.subplot(2, 1, 2)
        # plt.plot(space_y['pose'][28])
        # plt.axline((0, max(space_y['pose'][28]) * 0.25), (len(space_y['pose'][28]), max(space_y['pose'][28]) * 0.25))
        # plt.show()
        # endregion

        start_info = MATCH_FUNC.find_forward(time_y, point_sequence)  # score 2 and 3
        finish_info = MATCH_FUNC.find_backward(time_y, point_sequence)  # score 2 and 3
        # endregion

        # region score count
        # score1
        wrong = 0
        score_temp = []
        for model in movie_sequence.keys():
            for i in range(1, len(movie_sequence[model])):
                if movie_sequence[model][i - 1][0] == movie_sequence[model][i][0]:
                    wrong += 1
            score_temp.append((len(movie_sequence[model]) - wrong) / len(movie_sequence[model]) * 100)
        print(f"score1 {score_temp}")
        self.score[0] = min(0, np.array(score_temp).mean())
        # score2
        score_temp = []
        for model in space_y.keys():
            temp_array = []
            for point in space_y[model].keys():
                temp_array.append(
                    space_y[model][point][start_info[model][point][0] * 2: finish_info[model][point][0] * 2 + 3])
            min_len = min(map(len, temp_array))
            coef = np.corrcoef(temp_array[0][:min_len], temp_array[1][:min_len])[0, 1]
            score_temp.append(coef)
        print(f"score2 {score_temp}")
        self.score[1] = max(0, np.array(score_temp).mean() * 100)
        # score3
        score_temp = []
        for model in time_y.keys():
            temp_array = []
            for point in time_y[model].keys():
                temp_array.append(time_y[model][point][start_info[model][point][0]:finish_info[model][point][0]])
            min_len = min(map(len, temp_array))
            coef = np.corrcoef(temp_array[0][:min_len], temp_array[1][:min_len])[0, 1]
            score_temp.append(coef)
        print(f"score3 {score_temp}")
        self.score[2] = max(0, np.array(score_temp).mean() * 100)
        # endregion

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action11:
    def __init__(self, path):
        self.config = {'pose': [27, 28]}
        self.video_path = path
        self.score = [0 for _ in range(3)]

    def count_score(self, norm_data):
        # region feature_extraction
        space_y = MATCH_FUNC.space_position(norm_data, self.config, "x")
        time_y = MATCH_FUNC.time_speed(space_y)
        # endregion

        # region moving_side_judge
        movie_sequence = {}  # score 1
        point_sequence = {}
        for model in time_y.keys():
            movie_sequence[model] = []
            point_sequence[model] = {}
            for point in time_y[model].keys():
                point_sequence[model][point] = []
                min_moving_speed = max(time_y[model][point]) * 0.25
                find = False
                temp_info = None
                for i, value in enumerate(time_y[model][point]):
                    if value >= min_moving_speed and find == False:
                        find = True
                        temp_info = [point, i, value]
                    elif value >= min_moving_speed and find == True:
                        if value > temp_info[2]:
                            temp_info = [point, i, value]
                    elif value < min_moving_speed and find == True:
                        find = False
                        movie_sequence[model].append(temp_info)
                        point_sequence[model][point].append(temp_info)
                        temp_info = None

            # movie_sequence[[pos, point, value]]
            movie_sequence[model].sort(key=lambda x: x[1])

        # region draw
        # plt.subplot(2, 1, 1)
        # plt.plot(time_y['pose'][27])
        # plt.axline((0, max(time_y['pose'][27]) * 0.25), (len(time_y['pose'][27]), max(time_y['pose'][27]) * 0.25))
        # plt.subplot(2, 1, 2)
        # plt.plot(time_y['pose'][28])
        # plt.axline((0, max(time_y['pose'][28]) * 0.25), (len(time_y['pose'][28]), max(time_y['pose'][28]) * 0.25))
        # plt.show()
        # endregion

        start_info = MATCH_FUNC.find_forward(time_y, point_sequence)  # score 2 and 3
        finish_info = MATCH_FUNC.find_backward(time_y, point_sequence)  # score 2 and 3
        # endregion

        # region score count
        # score1
        wrong = 0
        score_temp = []
        print(movie_sequence)
        for model in movie_sequence.keys():
            for i in range(1, len(movie_sequence[model])):
                if movie_sequence[model][i - 1][0] == movie_sequence[model][i][0]  \
                    and movie_sequence[model][i][0] - movie_sequence[model][i - 1][0] > 10 :
                    wrong += 1
            score_temp.append((len(movie_sequence[model]) - wrong) / len(movie_sequence[model]) * 100)
        print(f"score1 {np.array(score_temp).mean()}")
        self.score[0] = max(0, np.array(score_temp).mean())
        # score2
        score_temp = []
        for model in space_y.keys():
            temp_array = []
            for point in space_y[model].keys():
                temp_array.append(
                    space_y[model][point][start_info[model][point][0] * 2: finish_info[model][point][0] * 2 + 3])
            min_len = min(map(len, temp_array))
            coef = np.corrcoef(temp_array[0][:min_len], temp_array[1][:min_len])[0, 1]
            score_temp.append(coef)
        print(f"score2 {np.array(score_temp).mean() * 100}")
        self.score[1] = max(0, np.array(score_temp).mean() * 100)
        # score3
        score_temp = []
        for model in time_y.keys():
            temp_array = []
            for point in time_y[model].keys():
                temp_array.append(time_y[model][point][start_info[model][point][0]:finish_info[model][point][0]])
            min_len = min(map(len, temp_array))
            coef = np.corrcoef(temp_array[0][:min_len], temp_array[1][:min_len])[0, 1]
            score_temp.append(coef)
        print(f"score3 {np.array(score_temp).mean() * 100}")
        self.score[2] = max(0, np.array(score_temp).mean() * 100)
        # endregion

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action12:
    def __init__(self, path):
        self.config = {'pose': [15, 16, 27, 28]}
        self.video_path = path
        self.score = [0 for _ in range(2)]

    def count_score(self, norm_data):
        # region feature_extraction
        space_y = MATCH_FUNC.space_position(norm_data, self.config, "y")
        time_y = MATCH_FUNC.time_speed(space_y)
        space_x = MATCH_FUNC.space_position(norm_data, self.config, "x")
        time_x = MATCH_FUNC.time_speed(space_x)
        # endregion

        # region score count
        # score1
        coef = abs(np.corrcoef(space_y["pose"][16], space_y["pose"][15])[0, 1])
        coef += abs(np.corrcoef(space_x["pose"][27], space_x["pose"][28])[0, 1])
        print(f"score1 {abs(coef / 2 * 100)}")
        self.score[0] = abs(coef / 2 * 100)
        # score2
        coef = abs(np.corrcoef(time_x["pose"][16], time_x["pose"][15])[0, 1])
        coef += abs(np.corrcoef(time_x["pose"][27], time_x["pose"][28])[0, 1])
        print(f"score2 {abs(coef / 2 * 100)}")
        self.score[1] = abs(coef / 2 * 100)
        # endregion

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action13:
    def __init__(self, path):
        self.config = {'pose': [15, 16]}
        self.video_path = path
        self.score = [0 for _ in range(2)]

    def count_score(self, norm_data):
        # region feature_extraction
        space_y = MATCH_FUNC.space_position(norm_data, self.config, "y")
        time_y = MATCH_FUNC.time_speed(space_y)
        space_x = MATCH_FUNC.space_position(norm_data, self.config, "x")
        time_x = MATCH_FUNC.time_speed(space_x)
        # endregion

        # region score count
        # score1
        coef = abs(np.corrcoef(space_y["pose"][16], space_y["pose"][15])[0, 1])
        coef += abs(np.corrcoef(space_x["pose"][16], space_x["pose"][15])[0, 1])
        print(f"score1 {abs(coef / 2* 100)}")
        self.score[0] = abs(coef / 2* 100)
        # score2
        coef = abs(np.corrcoef(time_x["pose"][16], time_x["pose"][15])[0, 1])
        coef += abs(np.corrcoef(time_y["pose"][16], time_y["pose"][15])[0, 1])
        print(f"score2 {abs(coef / 2 * 100)}")
        self.score[1] = abs(coef / 2 * 100)
        # endregion

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action14:
    def __init__(self, path):
        self.config = {'pose': [27, 28]}
        self.video_path = path
        self.score = [0 for _ in range(3)]

    def count_score(self, norm_data):
        # region feature_extraction
        space_y = MATCH_FUNC.space_position(norm_data, self.config, "y")
        time_y = MATCH_FUNC.time_speed(space_y)
        # endregion

        # region moving_side_judge
        movie_sequence = {}  # score 1
        point_sequence = {}
        for model in time_y.keys():
            movie_sequence[model] = []
            point_sequence[model] = {}
            for point in time_y[model].keys():
                point_sequence[model][point] = []
                min_moving_speed = max(time_y[model][point]) * 0.4
                find = False
                temp_info = None
                for i, value in enumerate(time_y[model][point]):
                    if value >= min_moving_speed and find == False:
                        find = True
                        temp_info = [point, i, value]
                    elif value >= min_moving_speed and find == True:
                        if value > temp_info[2]:
                            temp_info = [point, i, value]
                    elif value < min_moving_speed and find == True:
                        find = False
                        movie_sequence[model].append(temp_info)
                        point_sequence[model][point].append(temp_info)
                        temp_info = None

            # movie_sequence[[pos, point, value]]
            movie_sequence[model].sort(key=lambda x: x[1])

        # region draw
        # plt.subplot(2, 1, 1)
        # plt.plot(time_y['pose'][27])
        # plt.axline((0, max(time_y['pose'][27]) * 0.45), (len(time_y['pose'][27]), max(time_y['pose'][27]) * 0.4))
        # plt.subplot(2, 1, 2)
        # plt.plot(time_y['pose'][28])
        # plt.axline((0, max(time_y['pose'][28]) * 0.45), (len(time_y['pose'][28]), max(time_y['pose'][28]) * 0.4))
        # plt.show()
        # endregion

        start_info = MATCH_FUNC.find_forward(time_y, point_sequence)  # score 2 and 3
        finish_info = MATCH_FUNC.find_backward(time_y, point_sequence)  # score 2 and 3
        # endregion

        # region score count
        # score1
        wrong = 0
        score_temp = []
        for model in movie_sequence.keys():
            for i in range(1, len(movie_sequence[model])):
                if movie_sequence[model][i - 1][0] == movie_sequence[model][i][0]:
                    wrong += 1
            score_temp.append((len(movie_sequence[model]) - wrong) / len(movie_sequence[model]) * 100)
        print(f"score1 {score_temp}")
        self.score[0] = max(0, np.array(score_temp).mean())
        # score2
        score_temp = []
        for model in space_y.keys():
            temp_array = []
            for point in space_y[model].keys():
                temp_array.append(
                    space_y[model][point][start_info[model][point][0] * 2: finish_info[model][point][0] * 2 + 3])
            min_len = min(map(len, temp_array))
            coef = np.corrcoef(temp_array[0][:min_len], temp_array[1][:min_len])[0, 1]
            score_temp.append(coef)
        print(f"score2 {score_temp}")
        self.score[1] = max(0, np.array(score_temp).mean() * 100)
        # score3
        score_temp = []
        for model in time_y.keys():
            temp_array = []
            for point in time_y[model].keys():
                temp_array.append(time_y[model][point][start_info[model][point][0]:finish_info[model][point][0]])
            min_len = min(map(len, temp_array))
            coef = np.corrcoef(temp_array[0][:min_len], temp_array[1][:min_len])[0, 1]
            score_temp.append(coef)
        print(f"score3 {score_temp}")
        self.score[2] = max(0, np.array(score_temp).mean() * 100)
        # endregion

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

class Action15:
    def __init__(self, path):
        self.config = {'pose': [27, 28]}
        self.video_path = path
        self.score = [0 for _ in range(3)]

    def count_score(self, norm_data):
        # region feature_extraction
        space_x = MATCH_FUNC.space_position(norm_data, self.config, "x")
        time_x = MATCH_FUNC.time_speed(space_x)
        space_y = MATCH_FUNC.space_position(norm_data, self.config, "y")
        time_y = MATCH_FUNC.time_speed(space_y)
        # endregion

        # region moving_side_judge
        movie_sequence_y = {}  # score 1
        point_sequence_y = {}
        for model in time_y.keys():
            movie_sequence_y[model] = []
            point_sequence_y[model] = {}
            for point in time_y[model].keys():
                point_sequence_y[model][point] = []
                min_moving_speed = max(time_y[model][point]) * 0.3
                find = False
                temp_info = None
                for i, value in enumerate(time_y[model][point]):
                    if value >= min_moving_speed and find == False:
                        find = True
                        temp_info = [point, i, value]
                    elif value >= min_moving_speed and find == True:
                        if value > temp_info[2]:
                            temp_info = [point, i, value]
                    elif value < min_moving_speed and find == True:
                        find = False
                        movie_sequence_y[model].append(temp_info)
                        point_sequence_y[model][point].append(temp_info)
                        temp_info = None

            # movie_sequence[[pos, point, value]]
            movie_sequence_y[model].sort(key=lambda x: x[1])
        movie_sequence_x = {}  # score 1
        point_sequence_x = {}
        for model in time_x.keys():
            movie_sequence_x[model] = []
            point_sequence_x[model] = {}
            for point in time_x[model].keys():
                point_sequence_x[model][point] = []
                min_moving_speed = max(time_x[model][point]) * 0.3
                find = False
                temp_info = None
                for i, value in enumerate(time_x[model][point]):
                    if value >= min_moving_speed and find == False:
                        find = True
                        temp_info = [point, i, value]
                    elif value >= min_moving_speed and find == True:
                        if value > temp_info[2]:
                            temp_info = [point, i, value]
                    elif value < min_moving_speed and find == True:
                        find = False
                        movie_sequence_x[model].append(temp_info)
                        point_sequence_x[model][point].append(temp_info)
                        temp_info = None
            # movie_sequence[[pos, point, value]]
            movie_sequence_x[model].sort(key=lambda x: x[1])

        start_info_x = MATCH_FUNC.find_forward(time_x, point_sequence_x)  # score 2 and 3
        finish_info_x = MATCH_FUNC.find_backward(time_x, point_sequence_x)  # score 2 and 3
        start_info_y = MATCH_FUNC.find_forward(time_y, point_sequence_y)  # score 2 and 3
        finish_info_y = MATCH_FUNC.find_backward(time_y, point_sequence_y)  # score 2 and 3

        # region draw
        # plt.subplot(2, 1, 1)
        # plt.plot(time_x['pose'][27])
        # plt.axline((0, max(time_x['pose'][27]) *  0.3), (len(time_x['pose'][27]), max(time_x['pose'][27]) * 0.3))
        # plt.axline((start_info_x['pose'][27][0], -1), (start_info_x['pose'][27][0], 1))
        # plt.axline((finish_info_x['pose'][27][0], -1), (finish_info_x['pose'][27][0], 1))
        # plt.subplot(2, 1, 2)
        # plt.plot(time_x['pose'][28])
        # plt.axline((0, max(time_x['pose'][28]) *  0.3), (len(time_x['pose'][28]), max(time_x['pose'][28]) * 0.3))
        # plt.axline((start_info_x['pose'][28][0], -1), (start_info_x['pose'][28][0], 1))
        # plt.axline((finish_info_x['pose'][28][0], -1), (finish_info_x['pose'][28][0], 1))
        # plt.show()
        # endregion
        # endregion

        # region score count
        # score1
        wrong = 0
        score_temp = []
        for model in movie_sequence_x.keys():
            for i in range(1, len(movie_sequence_x[model])):
                if movie_sequence_x[model][i - 1][0] == movie_sequence_x[model][i][0]:
                    wrong += 1
            score_temp.append((len(movie_sequence_x[model]) - wrong) / len(movie_sequence_x[model]) * 100)
        for model in movie_sequence_y.keys():
            for i in range(1, len(movie_sequence_y[model])):
                if movie_sequence_y[model][i - 1][0] == movie_sequence_y[model][i][0]:
                    wrong += 1
            score_temp.append((len(movie_sequence_y[model]) - wrong) / len(movie_sequence_y[model]) * 100)
        print(f"score1 {np.array(score_temp).mean()}")
        self.score[0] = max(0, np.array(score_temp).mean())
        # score2
        score_temp = []
        coef = np.corrcoef(space_y["pose"][27], space_y["pose"][28])[0, 1]
        score_temp.append(abs(coef))
        coef = np.corrcoef(space_x["pose"][27], space_x["pose"][28])[0, 1]
        score_temp.append(abs(coef))
        print(f"score2 {(1 - np.array(score_temp).mean()) * 100}")
        self.score[1] = max(0, (1 - np.array(score_temp).mean()) * 100)
        # score3
        score_temp = []
        score_temp = []
        coef = np.corrcoef(time_x["pose"][27], time_x["pose"][28])[0, 1]
        score_temp.append(abs(coef))
        coef = np.corrcoef(time_y["pose"][27], time_y["pose"][28])[0, 1]
        score_temp.append(abs(coef))
        print(f"score3 {(1 - np.array(score_temp).mean()) * 100}")
        self.score[2] = max(0, (1 - np.array(score_temp).mean()) * 100)
        # endregion

    def main_func(self):
        mdp = MDP()
        norm_data = mdp.get_data(self.video_path, list(self.config.keys()))
        self.count_score(norm_data)

#測試用
if __name__ == "__main__":
    target_action = 9
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
