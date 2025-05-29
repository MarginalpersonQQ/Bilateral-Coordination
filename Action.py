import cv2
import numpy
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarkerOptions
from mediapipe.tasks.python.vision import HandLandmarkerOptions
import matplotlib.pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def gesture_normalize(pose): # Normalize pose landmark
    #The pose has 33 points, each point has 1 x coordinate
    #I only want to use points [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    def dis(k1, k2):  # distance of two point
        d = pow(((k1[1] - k2[1]) * (k1[1] - k2[1]) + (k1[0] - k2[0]) * (k1[0] - k2[0])), .5)
        return d
    #normalization
    unit = dis([pose[11].x, pose[11].y], [pose[23].x, pose[23].y])
    center = [(pose[11].x + pose[12].x) / 2, (pose[11].y + pose[12].y) / 2]
    for i in range(len(pose)):
        pose[i].x = (pose[i].x-center[0])/unit
        pose[i].y = (pose[i].y-center[1])/unit
    return pose

#Action 1 ~ 3 辨識部分相同
class Action1:
    class PeakDataStruct:
        def __init__(self, start=0.0, start_pos=0, end=0.0, end_pos=0, peak_max=0.0, peak_max_pos=0):
            self.start = start
            self.start_pos = start_pos
            self.end = end
            self.end_pos = end_pos
            self.peak_max = peak_max
            self.peak_max_pos = peak_max_pos

    class PosePoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def __init__(self, path):
        print("Action1 初始化")
        self.from_starting_position = 0
        self.points = [15, 16]
        self.back_to_starting_position = 1
        self.cut = 0.989
        self.current_min = 1000
        self.r_min = []
        self.min_position = []
        self.min_count_no = 0
        self.path = path
        self.row_data = None
        self.return_data = None
        self.score = [0 for _ in range(4)]

    def find_peak(self, data, forward_find=0, mean_offset=0.01):
        info = {}  # return information
        for point in self.points:
            peak_detect = False
            temp = []
            process_data = [data[i][point].y for i in range(len(data))]
            process_data = numpy.array(process_data)
            data_mean = process_data.mean() + mean_offset  # 0.01 為了避免不相干的動作突破門檻
            print(f"data_mean : {data_mean}")
            peak_recorder = None
            for frame in range(len(data) - forward_find):
                if not peak_detect and peak_recorder is None and data[frame + forward_find][point].y > data_mean:  # 偵測變化是否超過平均值
                    peak_detect = True
                    peak_recorder = self.PeakDataStruct(start=data[frame][point].y, start_pos=frame)
                elif peak_detect and peak_recorder is not None and data[frame][point].y < data_mean and frame > peak_recorder.start_pos + 5:
                    peak_detect = False
                    peak_recorder.end = data[frame + forward_find][point].y
                    peak_recorder.end_pos = frame + forward_find
                    temp.append(peak_recorder)
                    peak_recorder = None # 重置 peak_recorder
                if peak_detect:
                    # 確保只有在 peak_recorder 已正確初始化時才訪問它
                    try:
                        if data[frame][point].y > peak_recorder.peak_max:
                            peak_recorder.peak_max = data[frame][point].y
                            peak_recorder.peak_max_pos = frame
                    except TypeError:
                        print("Here is no peak_recorder")

            info[point] = temp
        return info

    def mdp_process(self):
        model_path = r'./model/pose_landmarker_full.task'
        base_options = mp.tasks.BaseOptions
        pose_landmarker = mp.tasks.vision.PoseLandmarker
        vision_running_mode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=model_path),
            running_mode=vision_running_mode.IMAGE)
        cap = cv2.VideoCapture(self.path)
        data = {}
        with pose_landmarker.create_from_options(options) as landmarker:
            count_image = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('No More Frame In Here.')
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                pose_landmarker_result = landmarker.detect(mp_image)
                if pose_landmarker_result.pose_landmarks and len(pose_landmarker_result.pose_landmarks) > 0:
                    result = pose_landmarker_result.pose_landmarks[0]
                    normal_result = gesture_normalize(result)
                    data[count_image] = {}
                    for i in self.points:
                        data[count_image][i] = self.PosePoint(normal_result[i].x, normal_result[i].y)
                    count_image += 1
                else:
                    print('copy previous point')
                    data[count_image] = {}
                    for i in self.points:
                        data[count_image][i] = data[count_image - 1][i]
                    count_image += 1
        return data

    def count_score(self):
        data = []
        peak_width = [] # 波的寬度
        two_peak_maximum_pos_gap = [] # 兩波峰的距離
        two_peak_distance = [] # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = [] #下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.points:
            for i in range(len(self.return_data[point])):
                data.append(self.return_data[point][i])
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

    def draw_plot(self):
        for point in self.points:
            temp = []
            for i in range(len(self.row_data)):
                temp.append(self.row_data[i][point].y)
            plt.plot(temp)
            plt.show()

    def main_function(self):
        row_data = self.mdp_process()
        self.row_data = row_data
        self.return_data = self.find_peak(data = row_data)
        for point in self.points:
            print(f"Point : {point}")
            for i in range(len(self.return_data[point])):
                print(f"start: {self.return_data[point][i].start}, start_pos:{self.return_data[point][i].start_pos}")
                print(f"end: {self.return_data[point][i].end}, end_pos:{self.return_data[point][i].end_pos}")
                print(f"peak_max: {self.return_data[point][i].peak_max}, Peak_max_pos:{self.return_data[point][i].peak_max_pos}")
                print("--------------------------------------------------------------")
        # 缺少分數計算部分
        self.count_score()
        # 畫圖
        # self.draw_plot()
        print("執行結束")
class Action2:
    class PosePoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    class DataStruct:
        def __init__(self, start_pos, end_pos, direction):
            self.start_pos = start_pos
            self.end_pos = end_pos
            self.direction = direction

    def __init__(self, path):
        print("Action2 初始化")
        self.from_starting_position = 0
        self.points = [15, 16]
        self.back_to_starting_position = 1
        self.cut = 0.989
        self.current_min = 1000
        self.r_min = []
        self.min_position = []
        self.min_count_no = 0
        self.path = path
        self.score = [0 for _ in range(4)]
    def mdp_process(self):
        model_path = r'./model/pose_landmarker_full.task'
        base_options = mp.tasks.BaseOptions
        pose_landmarker = mp.tasks.vision.PoseLandmarker
        vision_running_mode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=model_path),
            running_mode=vision_running_mode.IMAGE)
        cap = cv2.VideoCapture(self.path)
        data = {}
        with pose_landmarker.create_from_options(options) as landmarker:
            count_image = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('No More Frame In Here.')
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                pose_landmarker_result = landmarker.detect(mp_image)
                if pose_landmarker_result.pose_landmarks and len(pose_landmarker_result.pose_landmarks) > 0:
                    result = pose_landmarker_result.pose_landmarks[0]
                    normal_result = gesture_normalize(result)
                    data[count_image] = {}
                    for i in self.points:
                        data[count_image][i] = self.PosePoint(normal_result[i].x, normal_result[i].y)
                    count_image += 1
                else:
                    print('copy previous point')
                    data[count_image] = {}
                    for i in self.points:
                        data[count_image][i] = data[count_image - 1][i]
                    count_image += 1
        return data
    def find_direction(self, number1, number2):
        return 0 if number2 - number1 <= 0 else 1
    def find_peak(self, data):
        result = {15:[], 16:[]}
        for point in [15, 16]:
            length = 0
            temp_start_pos = 0
            temp_end_pos = 0
            direction = 0  # 0 negative, 1 positive
            true_data = False
            for frame in range(1, len(data)):
                if frame == 1:
                    direction = self.find_direction(data[frame-1][point].y, data[frame][point].y)
                if direction == self.find_direction(data[frame-1][point].y, data[frame][point].y):
                    length += 1
                    if abs(data[frame-1][point].y - data[frame][point].y) > 0.05:
                        true_data = True
                    if length == 3:
                        temp_start_pos = frame - length
                else:
                    if length >= 3 and true_data:
                        temp_end_pos = frame - 1
                        final = self.DataStruct(temp_start_pos, temp_end_pos, direction)
                        result[point].append(final)
                    length = 0
                    true_data = False
                    direction = self.find_direction(data[frame-1][point].y, data[frame][point].y)
                print(length, true_data)
        return result

    def count_score(self):
        pass
    def main_function(self):
        row_data = self.mdp_process()
        return_data = self.find_peak(data=row_data)
        for i in return_data[15]:
            print(i.start_pos, i.end_pos, i.direction)
        for i in return_data[16]:
            print(i.start_pos, i.end_pos, i.direction)
        print("執行結束")
class Action3:
    class PeakDataStruct:
        def __init__(self, start=0.0, start_pos=0, end=0.0, end_pos=0, peak_max=0.0, peak_max_pos=0):
            self.start = start
            self.start_pos = start_pos
            self.end = end
            self.end_pos = end_pos
            self.peak_max = peak_max
            self.peak_max_pos = peak_max_pos

    class PosePoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def __init__(self, path):
        print("Action1 初始化")
        self.from_starting_position = 0
        self.points = [15, 16]
        self.back_to_starting_position = 1
        self.cut = 0.989
        self.current_min = 1000
        self.r_min = []
        self.min_position = []
        self.min_count_no = 0
        self.path = path
        self.score = [0 for _ in range(4)]
    def find_peak(self, data, forward_find=5, mean_offset=0.01):
        info = {}  # return information
        for point in self.points:
            peak_detect = False
            temp = []
            process_data = [data[i][point].y for i in range(len(data))]
            process_data = numpy.array(process_data)
            data_mean = process_data.mean() + mean_offset  # 0.01 為了避免不相干的動作突破門檻
            print(f"data_mean : {data_mean}")

            for frame in range(len(data) - forward_find):
                if not peak_detect and data[frame + forward_find][point].y > data_mean:  # 偵測變化是否超過平均值
                    peak_detect = True
                    peak_recorder = self.PeakDataStruct(start=data[frame][point].y, start_pos=frame)
                elif peak_detect and data[frame][point].y < data_mean and frame > peak_recorder.start_pos + 5:
                    peak_detect = False
                    peak_recorder.end = data[frame + forward_find - 3][point].y
                    peak_recorder.end_pos = frame + forward_find - 3
                    temp.append(peak_recorder)
                    del peak_recorder  # 重置 peak_recorder
                if peak_detect:
                    # 確保只有在 peak_recorder 已正確初始化時才訪問它
                    try:
                        if data[frame][point].y > peak_recorder.peak_max:
                            peak_recorder.peak_max = data[frame][point].y
                            peak_recorder.peak_max_pos = frame
                    except TypeError:
                        print("Here is no peak_recorder")

            info[point] = temp
        return info

    def mdp_process(self):
        model_path = r'./model/pose_landmarker_full.task'
        base_options = mp.tasks.BaseOptions
        pose_landmarker = mp.tasks.vision.PoseLandmarker
        vision_running_mode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=model_path),
            running_mode=vision_running_mode.IMAGE)
        cap = cv2.VideoCapture(self.path)
        data = {}
        with pose_landmarker.create_from_options(options) as landmarker:
            count_image = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('No More Frame In Here.')
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                pose_landmarker_result = landmarker.detect(mp_image)
                if pose_landmarker_result.pose_landmarks and len(pose_landmarker_result.pose_landmarks) > 0:
                    result = pose_landmarker_result.pose_landmarks[0]
                    normal_result = gesture_normalize(result)
                    data[count_image] = {}
                    for i in self.points:
                        data[count_image][i] = self.PosePoint(normal_result[i].x, normal_result[i].y)
                    count_image += 1
                else:
                    print('copy previous point')
                    data[count_image] = {}
                    for i in self.points:
                        data[count_image][i] = data[count_image - 1][i]
                    count_image += 1
        return data

    def count_score(self):
        pass

    def main_function(self):
        row_data = self.mdp_process()
        return_data = self.find_peak(data=row_data)
        for point in self.points:
            print(f"Point : {point}")
            for i in range(len(return_data[point])):
                print(f"start: {return_data[point][i].start}, start_pos:{return_data[point][i].start_pos}")
                print(f"end: {return_data[point][i].end}, end_pos:{return_data[point][i].end_pos}")
                print(f"peak_max: {return_data[point][i].peak_max}, Peak_max_pos:{return_data[point][i].peak_max_pos}")
                print("--------------------------------------------------------------")
        # 缺少分數計算部分
        self.count_score()
        print("執行結束")
class Action4:
    #need hand landmark
    class PeakDataStruct:
        def __init__(self, start=0.0, start_pos=0, end=0.0, end_pos=0, peak_max=0.0, peak_max_pos=0):
            self.start = start
            self.start_pos = start_pos
            self.end = end
            self.end_pos = end_pos
            self.peak_max = peak_max
            self.peak_max_pos = peak_max_pos

    class PosePoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def __init__(self, path):
        print("Action1 初始化")
        self.from_starting_position = 0
        self.points = [15, 16]
        self.back_to_starting_position = 1
        self.cut = 0.989
        self.current_min = 1000
        self.r_min = []
        self.min_position = []
        self.min_count_no = 0
        self.path = path
        self.score = [0 for _ in range(4)]
    def find_peak(self, data, forward_find=5, mean_offset=0.01):
        info = {}  # return information
        for point in self.points:
            peak_detect = False
            temp = []
            process_data = [data[i][point].y for i in range(len(data))]
            process_data = numpy.array(process_data)
            data_mean = process_data.mean() + mean_offset  # 0.01 為了避免不相干的動作突破門檻
            print(f"data_mean : {data_mean}")

            for frame in range(len(data) - forward_find):
                if not peak_detect and data[frame + forward_find][point].y > data_mean:  # 偵測變化是否超過平均值
                    peak_detect = True
                    peak_recorder = self.PeakDataStruct(start=data[frame][point].y, start_pos=frame)
                elif peak_detect and data[frame][point].y < data_mean and frame > peak_recorder.start_pos + 5:
                    peak_detect = False
                    peak_recorder.end = data[frame + forward_find - 3][point].y
                    peak_recorder.end_pos = frame + forward_find - 3
                    temp.append(peak_recorder)
                    del peak_recorder  # 重置 peak_recorder
                if peak_detect:
                    # 確保只有在 peak_recorder 已正確初始化時才訪問它
                    try:
                        if data[frame][point].y > peak_recorder.peak_max:
                            peak_recorder.peak_max = data[frame][point].y
                            peak_recorder.peak_max_pos = frame
                    except TypeError:
                        print("Here is no peak_recorder")

            info[point] = temp
        return info

    def hand_normalize(self, hand, pose):
        def dis(k1, k2):  # distance of two point
            d = pow(((k1[1] - k2[1]) * (k1[1] - k2[1]) + (k1[0] - k2[0]) * (k1[0] - k2[0])), .5)
            return d
        # normalization
        unit = dis([pose[11].x, pose[11].y], [pose[23].x, pose[23].y])
        center = [(pose[11].x + pose[12].x) / 2, (pose[11].y + pose[12].y) / 2]
        for i in range(len(pose)):
            hand[i].x = (hand[i].x - center[0]) / unit
            hand[i].y = (hand[i].y - center[1]) / unit
        return hand

    def mdp_process(self):
        pose_model_path = r'./model/pose_landmarker_full.task'
        hand_model_path = r'./model/hand_landmarker.task'
        base_options = mp.tasks.BaseOptions
        pose_landmarker = mp.tasks.vision.PoseLandmarker
        hand_landmarker = mp.tasks.vision.HandLandmarker
        vision_running_mode = mp.tasks.vision.RunningMode
        options_pose = PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=pose_model_path),
            running_mode=vision_running_mode.IMAGE)
        options_hand = HandLandmarkerOptions(
            base_options=base_options(model_asset_path=hand_model_path),
            running_mode=vision_running_mode.IMAGE)
        cap = cv2.VideoCapture(self.path)
        pose_data = {}
        hand_data = {}
        with pose_landmarker.create_from_options(options_pose) as pose_land, hand_landmarker.create_from_options(options_hand) as hand_land:
            count_image = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('No More Frame In Here.')
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                pose_landmarker_result = pose_land.detect(mp_image)
                hand_landmarker_result = hand_land.detect(mp_image)
                #pose_landmark process block
                if pose_landmarker_result.pose_landmarks and len(pose_landmarker_result.pose_landmarks) > 0:
                    pose_result = pose_landmarker_result.pose_landmarks[0]
                    pose_normal_result = gesture_normalize(pose_result)
                    pose_data[count_image] = {}
                    for i in self.points:
                        pose_data[count_image][i] = self.PosePoint(pose_normal_result[i].x, pose_normal_result[i].y)
                    #--- hand_landmark process block ---
                    if hand_landmarker_result.hand_landmarks and len(hand_landmarker_result.hand_landmarks) > 0:
                        hand_result = pose_landmarker_result.pose_landmarks[0]
                        hand_normal_result = self.hand_normalize(hand_result, pose_result)
                        hand_data[count_image] = {}
                        for i in self.points:
                            hand_data[count_image][i] = self.PosePoint(hand_normal_result[i].x, hand_normal_result[i].y)
                    else:
                        print('copy previous hand point')
                        hand_data[count_image] = {}
                        for i in self.points:
                            hand_data[count_image][i] = hand_data[count_image - 1][i]
                else:
                    print('copy previous pose point')
                    pose_data[count_image] = {}
                    for i in self.points:
                        pose_data[count_image][i] = pose_data[count_image - 1][i]
                count_image += 1
        return pose_data, hand_data

    def count_score(self):
        pass

    def main_function(self):
        row_pose_data, row_hand_data = self.mdp_process()
        return_data = self.find_peak(data=row_pose_data)
        for point in self.points:
            print(f"Point : {point}")
            for i in range(len(return_data[point])):
                print(f"start: {return_data[point][i].start}, start_pos:{return_data[point][i].start_pos}")
                print(f"end: {return_data[point][i].end}, end_pos:{return_data[point][i].end_pos}")
                print(f"peak_max: {return_data[point][i].peak_max}, Peak_max_pos:{return_data[point][i].peak_max_pos}")
                print("--------------------------------------------------------------")
        # 缺少分數計算部分
        self.count_score()
        print("執行結束")
class Action5:
    class PeakDataStruct:
        def __init__(self, start=0.0, start_pos=0, end=0.0, end_pos=0, peak_max=0.0, peak_max_pos=0):
            self.start = start
            self.start_pos = start_pos
            self.end = end
            self.end_pos = end_pos
            self.peak_max = peak_max
            self.peak_max_pos = peak_max_pos

    class PosePoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def __init__(self, path):
        print("Action1 初始化")
        self.from_starting_position = 0
        self.points = [15, 16]
        self.back_to_starting_position = 1
        self.cut = 0.989
        self.current_min = 1000
        self.r_min = []
        self.min_position = []
        self.min_count_no = 0
        self.path = path
        self.row_data = None
        self.return_data = None
        self.score = [0 for _ in range(4)]

    def find_peak(self, data, forward_find=0, mean_offset=0.01):
        info = {}  # return information
        for point in self.points:
            peak_detect = False
            temp = []
            process_data = [data[i][point].y for i in range(len(data))]
            process_data = numpy.array(process_data)
            data_mean = process_data.mean() + mean_offset  # 0.01 為了避免不相干的動作突破門檻
            print(f"data_mean : {data_mean}")
            peak_recorder = None
            for frame in range(len(data) - forward_find):
                if not peak_detect and peak_recorder is None and data[frame + forward_find][
                    point].y > data_mean:  # 偵測變化是否超過平均值
                    peak_detect = True
                    peak_recorder = self.PeakDataStruct(start=data[frame][point].y, start_pos=frame)
                elif peak_detect and peak_recorder is not None and data[frame][
                    point].y < data_mean and frame > peak_recorder.start_pos + 5:
                    peak_detect = False
                    peak_recorder.end = data[frame + forward_find][point].y
                    peak_recorder.end_pos = frame + forward_find
                    temp.append(peak_recorder)
                    peak_recorder = None  # 重置 peak_recorder
                if peak_detect:
                    # 確保只有在 peak_recorder 已正確初始化時才訪問它
                    try:
                        if data[frame][point].y > peak_recorder.peak_max:
                            peak_recorder.peak_max = data[frame][point].y
                            peak_recorder.peak_max_pos = frame
                    except TypeError:
                        print("Here is no peak_recorder")

            info[point] = temp
        return info

    def mdp_process(self):
        model_path = r'./model/pose_landmarker_full.task'
        base_options = mp.tasks.BaseOptions
        pose_landmarker = mp.tasks.vision.PoseLandmarker
        vision_running_mode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=model_path),
            running_mode=vision_running_mode.IMAGE)
        cap = cv2.VideoCapture(self.path)
        data = {}
        with pose_landmarker.create_from_options(options) as landmarker:
            count_image = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('No More Frame In Here.')
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                pose_landmarker_result = landmarker.detect(mp_image)
                if pose_landmarker_result.pose_landmarks and len(pose_landmarker_result.pose_landmarks) > 0:
                    result = pose_landmarker_result.pose_landmarks[0]
                    normal_result = gesture_normalize(result)
                    data[count_image] = {}
                    for i in self.points:
                        data[count_image][i] = self.PosePoint(normal_result[i].x, normal_result[i].y)
                    count_image += 1
                else:
                    print('copy previous point')
                    data[count_image] = {}
                    for i in self.points:
                        data[count_image][i] = data[count_image - 1][i]
                    count_image += 1
        return data

    def count_score(self):
        data = []
        peak_width = []  # 波的寬度
        two_peak_maximum_pos_gap = []  # 兩波峰的距離
        two_peak_distance = []  # 前一個波的結束到下一個波的開始的距離
        st_to_max_to_end_diff = []  # 下去 -> 上來的時間差距
        num_of_peak = 6
        score = 0
        for point in self.points:
            for i in range(len(self.return_data[point])):
                data.append(self.return_data[point][i])
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

    def draw_plot(self):
        for point in self.points:
            temp = []
            for i in range(len(self.row_data)):
                temp.append(self.row_data[i][point].y)
            plt.plot(temp)
            plt.show()

    def main_function(self):
        row_data = self.mdp_process()
        self.row_data = row_data
        self.return_data = self.find_peak(data=row_data)
        for point in self.points:
            print(f"Point : {point}")
            for i in range(len(self.return_data[point])):
                print(f"start: {self.return_data[point][i].start}, start_pos:{self.return_data[point][i].start_pos}")
                print(f"end: {self.return_data[point][i].end}, end_pos:{self.return_data[point][i].end_pos}")
                print(
                    f"peak_max: {self.return_data[point][i].peak_max}, Peak_max_pos:{self.return_data[point][i].peak_max_pos}")
                print("--------------------------------------------------------------")
        # 缺少分數計算部分
        self.count_score()
        # 畫圖
        # self.draw_plot()
        print("執行結束")
class Action6:
    #有需要手指嗎?
    def __init__(self):
        self.landmark = [15, 16]
class Action8:
    def __init__(self):
        self.landmark = [15, 16]
class Action9:
    def __init__(self):
        self.landmark = [15, 16]
class Action10:
    def __init__(self):
        self.landmark = [15, 16]
class Action11:
    def __init__(self):
        self.landmark = [15, 16]
class Action12:
    def __init__(self):
        self.landmark = [15, 16]