import tkinter as tk
from tkinter import filedialog, messagebox
import Action3_0
import threading
import os
import datetime
from openpyxl import Workbook, load_workbook

file_var = None
score1 = None
score2 = None
score3 = None
score4 = None
score5 = None
video_fold_root_path = r"C:\Bilateral Coordination Record Video"
video_slots = None

excel_lock = threading.Lock()
start_time_str = None  # 記錄本次判斷的時間字串
amount_of_scores = 1
amount_of_actions = 15

def init_excel_single_sheet():
    global amount_of_scores, amount_of_actions
    """建立單一工作表，格式如使用者提供的圖片。"""
    excel_path = os.path.join(video_fold_root_path, "scores.xlsx")

    if not os.path.exists(excel_path):
        wb = Workbook()
        ws = wb.active
        ws.title = "Scores"

        # 第一列：影片標題 A1 A1 A1 A1 A2 A2 A2 A2 ...
        titles = ["動作"]
        for i in range(1, 16):
            titles += [f"Action {i}"] * amount_of_scores
        ws.append(titles)

        # 第二列：分數1 分數2 分數3 分數4 ...
        score_names = ["名稱"]
        for i in range(amount_of_actions):
            score_names += [f"得分{i}"]
        ws.append(score_names)

        wb.save(excel_path)

    return excel_path

def save_scores_to_excel(video_name, index, scores):
    global start_time_str, amount_of_scores, amount_of_actions

    excel_path = init_excel_single_sheet()  # 新的初始化（下面會給）

    with excel_lock:
        wb = load_workbook(excel_path)
        ws = wb.active

        # 若 scores 不足 4 個 -> 補 -1
        fixed_scores = []
        for i in range(amount_of_scores):
            if i < len(scores) and isinstance(scores[i], (int, float)):
                fixed_scores.append(scores[i])
            else:
                fixed_scores.append(-1)

        # 尋找名稱是否已存在 -> 更新同一列
        target_row = None
        for row in range(3, ws.max_row + 1):
            if ws.cell(row=row, column=1).value == video_name:
                target_row = row
                break

        # 若沒有 -> 新增一列
        if target_row is None:
            target_row = ws.max_row + 1
            ws.cell(row=target_row, column=1, value=video_name)

        # 計算目標欄位：每個影片占 4 欄
        # index=0 → A1 分數1~4 → column 2~5
        # index=1 → A2 分數1~4 → column 6~9
        start_col = index * amount_of_scores + 2

        for i in range(amount_of_scores):
            ws.cell(row=target_row, column=start_col + i, value=fixed_scores[i])

        wb.save(excel_path)

def start_action():
    global start_time_str
    video_path = file_var.get()

    # 本次判斷時間 (所有 index 共用)
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    messagebox.showinfo("開始判斷", f"開始處理影片：{video_path}")
    run_all_actions(video_path)

def update_grid(index, result=None, not_found=False):
    global video_slots

    if not_found:
        for label in video_slots[index]:
            label.config(text="No Score")
        save_scores_to_excel(file_var.get(), index, ["No Score"]*amount_of_scores)
    else:
        scores = result.score
        scores_len = len(scores)

        # 更新UI
        for i, label in enumerate(video_slots[index]):
            if i < scores_len:
                label.config(text=str(int(scores[i])))
            else:
                label.config(text="No Score")

        # 寫入 Excel
        save_scores_to_excel(file_var.get(), index, scores)

def run_all_actions(video_path):
    global video_fold_root_path
    file_exten = ".mp4"
    filenames = [f"{i:02}{file_exten}" for i in range(1, 16)]


    def worker(index, filename):
        full_path = os.path.join(video_fold_root_path, video_path, filename)
        if not os.path.isfile(full_path):
            update_grid(index, not_found=True)
            return

        # 動態取得類別名，例如 Action1、Action2...Action15
        action_class_name = f"Action{index + 1}"
        action_class = getattr(Action3_0, action_class_name, None)

        if action_class is None:
            update_grid(index, not_found=True)
            print(f"找不到對應類別: {action_class_name}")
            return


        result = action_class(full_path)
        result.main_func()
        update_grid(index, result)

    for i, name in enumerate(filenames):
        threading.Thread(target=worker, args=(i, name)).start()
def clear_all_slots():
    global video_slots
    if video_slots is None:
        return
    for slot_labels in video_slots:
        for label in slot_labels:
            label.config(text="")
# 動作處理函數
def start_action():
    global start_time_str
    global file_var
    global score1, score2, score3, score4, score5
    global video_slots
    global video_fold_root_path

    video_path = file_var.get()

    # 設定本次起始時間
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 清空上一次的顯示結果
    clear_all_slots()

    messagebox.showinfo("開始判斷", f"開始處理影片：{video_path}")

    # 啟動所有動作分析
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
    root.geometry("1024x512")

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
    headers = ["分數"]

    main_frame = tk.Frame(root)
    main_frame.pack()

    video_slots = []  # 用來記錄每個影片格子的 Label (5 個)

    for r in range(rows):
        for c in range(cols):
            if frame_index >= frame_count:
                break

            video_frame = tk.LabelFrame(main_frame, text=f"動作 {frame_index + 1}", padx=5, pady=5)
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
