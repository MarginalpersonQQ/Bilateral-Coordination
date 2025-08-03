import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class PalmOrientationDecide:

    def __init__(self, show_and_save_image = False):
        self.show_and_save_image = show_and_save_image
        self.std_threshold = 1.25
        plt.rcParams['font.family'] = 'Microsoft JhengHei'

    @staticmethod
    def linear_interpolate(data):
        nans = np.isnan(data)
        x = np.arange(len(data))
        data[nans] = np.interp(x[nans], x[~nans], data[~nans])
        return data

    def threshold_filter(self, data):
        threshold = np.nanmean(data) + np.nanstd(data) * self.std_threshold
        outlier_mask = data > threshold
        adjusted_data = data.copy()
        adjusted_data[outlier_mask] = np.nan
        return [adjusted_data, threshold]

    def avg_abs_process(self, data):
        data_mean = np.nanmean(data)
        data_diff = np.abs(data - data_mean)
        return data_diff

    def palm_orientation(self, data):
        def count_delta(data):
            delta = []
            for x1, x2 in zip(data["point4"]["x"], data["point20"]["x"]):
                print(f"4: {x1}, 20:{x2}")
                delta.append(x1 - x2)
            delta = np.array(delta)
            # delta_mean = delta.mean()
            # delta_std = delta.std()
            # delta_avg_0 = delta - delta_mean
            # print(f"mean: {delta_mean}")
            # print(f"std: {delta_std}")
            plt.title("測試")
            plt.plot(data["point4"]["x"], linestyle='--')
            plt.plot(data["point20"]["x"], linestyle='-.')
            plt.plot(delta)
            plt.show()
            plt.close()
            return delta

        return count_delta(data)



    def main_function(self, my_dict):
        results = {}
        for key, data in my_dict.items():
            print(key)
            results[key] = {}
            for data_key, data_data in data.items():
                if data_key == "x":
                    data_data_diff = self.avg_abs_process(data_data.copy())
                    data_data_threshold, threashold = self.threshold_filter(data_data_diff)
                    data_data_interpo = self.linear_interpolate(data_data_threshold.copy())

                    plt.subplot(2, 2, 1)
                    plt.title(f"原始數據 {key} {data_key}")
                    plt.plot(data_data)
                    plt.subplot(2, 2, 2)
                    plt.title("平均原點+絕對值")
                    plt.plot(data_data_diff)
                    plt.plot([threashold for _ in range(len(data_data))])
                    plt.subplot(2, 2, 3)
                    plt.title("閥值過率")
                    plt.plot(np.ma.masked_invalid(data_data_threshold))
                    plt.subplot(2, 2, 4)
                    plt.title("線性插值")
                    plt.plot(data_data_interpo)
                    plt.tight_layout()
                    plt.savefig(f"{key}_{data_key}.png")
                    plt.close()
                    results[key]['x'] = data_data_interpo.copy()

                if data_key == "y":
                    data_data_interpo = self.linear_interpolate(data_data.copy())
                    plt.subplot(1, 2, 1)
                    plt.title(f"原始數據 {key} {data_key}")
                    plt.plot(data_data)
                    plt.subplot(1, 2, 2)
                    plt.title("線性插值")
                    plt.plot(data_data_interpo)
                    plt.savefig(f"{key}_{data_key}.png")
                    plt.close()
                    results[key]['y'] = data_data_interpo.copy()

        return results

if __name__ == "__main__":
    my_dict = {}
    with  open("right_hand_landmark.txt", "r") as f:
        point = None
        cod = ""
        for lines, data in enumerate(f.readlines()):
            text = data.split()[0]
            if text[:5] == "point":
                point = text.split(":")[0]
                my_dict[point] = {}
            elif text[:1] in ["x", "y"]:
                cor = text.split(":")[0]
                my_dict[point][cor] = []
            elif point is not None and cor in ['x', 'y']:
                text = data.split()
                my_dict[point][cor] = np.array(list(map(float, text)))


    detect_hands = PalmOrientationDecide()
    processed_data = detect_hands.main_function(my_dict)
    detect_hands.palm_orientation(processed_data.copy())



