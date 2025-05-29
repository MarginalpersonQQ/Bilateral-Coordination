import tkinter as tk
from tkinter import filedialog, messagebox
import Action
import threading
import os

file_var = None
score1 = None
score2 = None
score3 = None
score4 = None
score5 = None
video_fold_root_path = r"C:\Vision Record Video"
video_slots = None


def update_grid(index, result=None, not_found=False):
    if not_found:
        for label in video_slots[index]:
            label.config(text="Not video")
    else:
        scores = result.score
        total = sum(scores)
        values = [scores[0], scores[1], scores[2], scores[3], total if total != 99 else 100]

        for i, label in enumerate(video_slots[index]):
            label.config(text=str(values[i]))

def run_all_actions(video_path):
    global video_fold_root_path
    filenames = [
        "01.mp4", "02.mp4", "03.mp4", "04.mp4", "05.mp4",
        "06.mp4", "07.mp4", "08.mp4", "09.mp4", "10.mp4",
        "11.mp4", "12.mp4", "13.mp4", "14.mp4", "15.mp4"
    ]


    def worker(index, filename):
        full_path = os.path.join(video_fold_root_path, video_path, filename)
        if not os.path.isfile(full_path):
            update_grid(index, not_found=True)
            return

        # 動態取得類別名，例如 Action1、Action2...Action15
        action_class_name = f"Action{index + 1}"
        action_class = getattr(Action, action_class_name, None)
        print(action_class)

        if action_class is None:
            update_grid(index, not_found=True)
            print(f"找不到對應類別: {action_class_name}")
            return

        result = action_class(full_path)
        result.main_function()
        update_grid(index, result)

    for i, name in enumerate(filenames):
        threading.Thread(target=worker, args=(i, name)).start()

# 動作處理函數
def start_action():
    global file_var
    global score1, score2, score3, score4, score5
    global video_slots
    global video_fold_root_path

    video_path = file_var.get()

    # 模擬處理程序（替換成你的動作識別處理邏輯）
    messagebox.showinfo("開始判斷", f"開始處理影片：{video_path}")
    run_all_actions(video_path)

def get_video_files():

    if not os.path.exists(video_fold_root_path):
        return []
    return [f for f in os.listdir(video_fold_root_path)]

def judge_init():
    global file_var
    global score1, score2, score3, score4, score5
    global video_slots


    print("子 UI 初始化")
    # 主視窗設定
    root = tk.Tk()
    root.title("動作判斷系統")
    root.geometry("1024x1024")

    # 影片選擇區
    tk.Label(root, text="選擇影片檔案：", font=("Arial", 18, "bold")).pack(pady=10)
    file_var = tk.StringVar()
    video_files = get_video_files()
    if video_files:
        file_var.set(video_files[0])  # 預設第一個影片
    else:
        file_var.set("（找不到影片）")
        print("No file path")
    tk.OptionMenu(root, file_var, *video_files).pack(pady=5)

    score1 = tk.StringVar()
    score2 = tk.StringVar()
    score3 = tk.StringVar()
    score4 = tk.StringVar()
    score5 = tk.StringVar()

    # 開始按鈕
    tk.Button(root, text="開始判斷", command=start_action, bg="green", fg="white").pack(pady=20)

    frame_count = 15
    cols = 5
    rows = 3
    frame_index = 0

    # 標題名稱
    headers = ["完整度", "穩定度", "連續性", "流暢度", "總分"]

    main_frame = tk.Frame(root)
    main_frame.pack()

    video_slots = []  # 用來記錄每個影片格子的 Label (5 個)

    for r in range(rows):
        for c in range(cols):
            if frame_index >= frame_count:
                break

            video_frame = tk.LabelFrame(main_frame, text=f"影片 {frame_index + 1}", padx=5, pady=5)
            video_frame.grid(row=r, column=c, padx=10, pady=10)

            slot_labels = []  # 儲存這個影片格子的5個欄位
            for i, header in enumerate(headers):
                label = tk.Label(video_frame, text=header, font=("Arial", 9, "bold"))
                label.grid(row=i, column=0, padx=3, pady=2)

                slot = tk.Label(video_frame, text="", relief="ridge", width=8, height=2)
                slot.grid(row=i, column=1, padx=3, pady=2)
                slot_labels.append(slot)

            video_slots.append(slot_labels)  # 存進影片格子的總表
            frame_index += 1

    # 主迴圈
    root.mainloop()


if __name__ == "__main__":
    judge_init()
