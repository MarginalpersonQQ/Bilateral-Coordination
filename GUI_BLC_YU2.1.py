import os
import cv2
import time
import pygame
import threading
import subprocess
import numpy as np
import moviepy as mp
import UI
import pyrealsense2 as rs
from pynput import keyboard
from datetime import datetime
import dearpygui.dearpygui as dpg
from screeninfo import get_monitors

"""Variable Declare"""
#region Variable Declare
screens = None
screen_count = None
main_screen = None
second_screen = None
MS_video_window_width = None
MS_video_window_height = None
video_path = None
out_file = None
save_root_path = None
save_case_path = None
case_name = None
case_name_set = None
video_fps = None
padding = None
pause = None
video_play_number = None
video_container = None
camera = None
demo_video = None
play_over = None
process = None
frame_size = None
start_play = None
start_record = None
record_over = None
camera_is_open = None
hidden_theme = None
warning_string = None
stop_event = None
press_control_video_button_theme = None
unpress_control_video_button_theme = None
thread = []
thread_video = []
frame_count = None
frame_now = None
record_time = None
start_record_time = None
record_frame_count = None
listen_keyboard = None
keyboard_listener_obj = None
judge_program_status = None
judge_program_running = None
loaded = None
v_frame = None
video_count = None
show_hint = None
#endregion

def init_parameter():
    global screens, screen_count
    global main_screen, second_screen
    global MS_video_window_width, MS_video_window_height
    global padding
    global video_path, save_root_path, out_file, video_count
    global fourcc, video_fps
    global case_name, case_name_set
    global pause
    global video_play_number
    global camera, demo_video
    global play_over, start_play
    global process, frame_size
    global start_record, record_over
    global camera_is_open
    global stop_event
    global record_time
    global record_frame_count
    global judge_program_running
    global loaded, v_frame
    global show_hint

    """Screen Setting"""
    screens = get_monitors()
    screen_count = len(screens)
    if screen_count < 2:
        main_screen = second_screen = screens[0]
    elif screens[0].is_primary:
        main_screen, second_screen = screens[0], screens[1]
    else:
        main_screen, second_screen = screens[1], screens[0]
    print(main_screen.width, main_screen.height)
    """Video Setting"""
    video_path = "video_input"
    save_root_path = r"C:\Bilateral Coordination Record Video"
    if not os.path.exists(video_path):
        print("Demo Video Path Not Found.")
        return
    """Video Play Number"""
    video_play_number = 0  # Start from video 0
    video_count = 16  # 0 ~ 20
    """Video Var Setting"""
    play_over = False
    start_play = False
    """Video Record"""
    fourcc = "MP4V"
    video_fps = 60
    """Video Loader"""
    loaded = False
    v_frame = []
    """Show Hing Flag"""
    show_hint = False
    """Judge Program Flag"""
    judge_program_running = False
    """Pygame Init"""
    pygame.mixer.init()
    """Threading Setting"""
    stop_event = threading.Event()
    """Recrod Setting"""
    start_record = False
    record_over = False
    record_frame_count = 0
    """Camera Setting"""
    camera_is_open = False
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    """Case Name And Pause Setting"""
    pause = True
    case_name = None
    case_name_set = False
    """Case Setting"""
    os.makedirs(save_root_path, exist_ok=True)
    """Main Screen Video Player Setting"""
    MS_video_window_height = 480
    MS_video_window_width = 640
    """Padding Setting"""
    padding = (main_screen.width - MS_video_window_width * 2) / 6

def dearpygui_setup():
    global main_screen, second_screen
    global padding
    global MS_video_window_height, MS_video_window_width
    global hidden_theme, warning_string
    global unpress_control_video_button_theme, press_control_video_button_theme
    global video_count
    global show_hint

    def set_show_hint_false():
        global show_hint
        show_hint = True

    dpg.create_context()
    """Window Setup"""
    # Hint window
    with dpg.window(label="流程提示", modal=True, show=False, tag="Hint_window", width=500, height=300,
                    pos=(main_screen.x // 2 - 250, main_screen.y // 2 - 150), no_close=True):
        dpg.add_text("", tag="hint_text")
        dpg.add_button(label="繼續",
                       callback=lambda _: (dpg.configure_item("Hint_window", show=False), set_show_hint_false()))
    # Alert window
    with dpg.window(label="繼續----->?", modal=True, show=False, tag="alert_window", width=300, height=150,
                    pos=(main_screen.x // 2 - 150, main_screen.y // 2 - 75), no_close=True):
        dpg.add_text("重新或繼續錄製?", tag="alert_text")
        dpg.add_button(label="重新", tag="go previous", callback=go_next_video)
        dpg.add_button(label="繼續", tag="go next", callback=go_next_video)
    # Main window
    with dpg.window(width=main_screen.width, height=main_screen.height, tag="main_window",
                    pos=(main_screen.x, main_screen.y), no_close=True, no_move=True, no_title_bar=True,
                    no_resize=True) as window:
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(640, 480, [0 for _ in range(640 * 480)], tag="LIVE_TEXTURE",
                                format=dpg.mvFormat_Float_rgb)
            dpg.add_raw_texture(640, 480, [0 for _ in range(640 * 480)], tag="DEMS_TEXTURE",
                                format=dpg.mvFormat_Float_rgb)
        with dpg.child_window(label="Demonstrator Video", width=MS_video_window_width,
                              height=MS_video_window_height,
                              pos=(padding * 2, padding), tag="DEMSWIN", no_scrollbar=True):
            dpg.add_image("DEMS_TEXTURE")
        with dpg.child_window(label="video_loading_text_box", width=MS_video_window_width,
                              height=padding,
                              pos=(padding * 2, padding + MS_video_window_height), tag="video_loading_box",
                              no_scrollbar=True):
            dpg.add_text("\n\n\t影片載入中...", tag="video_loading_hint", show=False)
        with dpg.child_window(label="Live Video", width=MS_video_window_width, height=MS_video_window_height,
                              pos=(padding * 4 + MS_video_window_width, padding), tag="LIVEWIN", no_scrollbar=True):
            dpg.add_image("LIVE_TEXTURE")
            with dpg.viewport_drawlist():
                dpg.draw_circle(
                    center=(padding * 4 + MS_video_window_width * 1.5, padding * 1.5 + MS_video_window_height),
                    radius=20, color=(255, 0, 0, 255), show=False, fill=(255, 0, 0, 125), tag="REC_ICON")
        with dpg.child_window(width=main_screen.width,
                              height=main_screen.height - (padding * 2 + MS_video_window_height + 60),
                              pos=(0, padding * 2 + MS_video_window_height), tag="function"):
            fwidth = dpg.get_item_width("function")
            fheight = dpg.get_item_height("function")
            with dpg.group(tag='button_group'):
                dpg.add_button(label="開始",
                               tag="Start Button",
                               pos=(int(padding * 2 + MS_video_window_width * 0.5 - fwidth // 15 * 3 / 2),
                                    fheight // 5 * 1),
                               width=fwidth // 15 * 3,
                               height=fheight // 5 * 1,
                               callback=button_listener)
                dpg.add_button(label="分數判斷",
                               tag="Judge Score",
                               pos=(int(padding * 2 + MS_video_window_width + padding * 0.2), fheight // 5 * 1),
                               width=fwidth // 15 * 1,
                               height=fheight // 5 * 1,
                               callback=button_listener,
                               show=True)
                dpg.add_button(label="結束錄影",
                               tag="End Record",
                               pos=(int(padding * 2 + MS_video_window_width + padding * 0.2), fheight // 5 * 3),
                               width=fwidth // 15 * 1,
                               height=fheight // 5 * 1,
                               callback=button_listener,
                               show=False)
            with dpg.group(horizontal=True):
                raw = 1
                for i in range(1, video_count):
                    temptag = '0' + str(i) if i < 10 else str(i)
                    if i > 15:
                        raw = 2
                    dpg.add_button(label=temptag,
                                   tag=temptag + '_video_button',
                                   width=int(MS_video_window_width / 15),
                                   height=int(MS_video_window_width / 15),
                                   pos=(
                                       int(padding * 4 + MS_video_window_width + int(MS_video_window_width / 15) * (
                                               i - (raw - 1) * 15)), fheight // 10 * raw),
                                   )
            with dpg.group(pos=(padding * 4 + MS_video_window_width, fheight // 10 * 5)):
                dpg.add_text("PLEASE ENTER CASE NAME FIRST", tag="warn_text")
                with dpg.group(horizontal=True):
                    input_box = dpg.add_input_text(tag='case_name_box', width=int(MS_video_window_width),
                                                   hint="USER NAME")
                    dpg.add_button(label="OK", callback=set_case_name, user_data=input_box, tag="case_name_button")
    """Item Style"""
    with dpg.theme() as loading_box_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255), category=dpg.mvThemeCat_Core)  # 白底
            dpg.add_theme_color(dpg.mvThemeCol_Border, (255, 255, 255), category=dpg.mvThemeCat_Core)  # 邊框白（等於隱藏）
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 0, category=dpg.mvThemeCat_Core)  # 邊框寬度為 0
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 0, category=dpg.mvThemeCat_Core)  # 無圓角
    with dpg.theme() as window_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (255, 255, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 9, category=dpg.mvThemeCat_Core)
    with dpg.theme() as button_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (232, 232, 232), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (138, 192, 218), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Text, (70, 70, 70), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 20, 20, category=dpg.mvThemeCat_Core)
    with dpg.theme() as function_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (70, 70, 70), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 20, category=dpg.mvThemeCat_Core)
    with dpg.theme() as press_control_video_button_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (245, 221, 37))
    with dpg.theme() as unpress_control_video_button_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (168, 168, 168))
    with dpg.theme() as warning_string:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0, 0), category=dpg.mvThemeCat_Core)
    with dpg.theme() as hidden_theme:
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0, 0))  # 設為完全透明
    # add a font registry
    with dpg.font_registry():
        with dpg.font(r"Font/jf-openhuninn-2.0.ttf", 20) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
            dpg.add_font_range(0x300, 0x400)
            dpg.bind_font(default_font)
    # Bind Sytle
    dpg.bind_item_theme("Start Button", button_theme)
    dpg.bind_item_theme("Judge Score", button_theme)
    dpg.bind_item_theme("End Record", button_theme)
    dpg.bind_item_theme("main_window", window_theme)
    dpg.bind_item_theme("function", function_theme)
    dpg.bind_item_theme("warn_text", warning_string)
    dpg.bind_item_theme("video_loading_hint", warning_string)
    dpg.bind_item_theme("video_loading_box", loading_box_theme)

    for i in range(1, video_count):
        tg = '0' + str(i) if i < 10 else str(i)
        dpg.bind_item_theme(tg + "_video_button", unpress_control_video_button_theme)
    """Create Window"""
    dpg.create_viewport(title="Bilateral Coordination", width=main_screen.width,
                        height=main_screen.height, x_pos=main_screen.x,
                        y_pos=main_screen.y)
    dpg.setup_dearpygui()
    dpg.show_viewport()

def reset_to_start():
    global start_record, record_over
    global play_over, start_play
    global video_play_number
    global pause
    global case_name_set
    global demo_video
    global stop_event, thread, out_file
    global frame_now, frame_count
    global  record_frame_count
    global loaded, v_frame
    global show_hint

    show_hint = False
    loaded = False
    v_frame = []
    record_frame_count = 0
    frame_now = 0
    frame_count = None
    start_record = False
    play_over = False
    start_play = False
    record_over = False
    video_play_number = 0
    case_name_set = False
    demo_video = None
    out_file = None
    try:
        pygame.mixer.music.unload()
        pygame.mixer.quit()
    except:
        pass
    dpg.bind_item_theme("warn_text", warning_string)
    dpg.configure_item('case_name_box', enabled=True)
    dpg.set_item_label("case_name_button", "Ok")
    dpg.set_item_label("Start Button", "開始")
    # dpg.configure_item("Record Previous Again", show=False)
    dpg.configure_item("End Record", show=False)
    dpg.configure_item("REC_ICON", show=False)
    dpg.configure_item("alert_window", show=False)
    dpg.configure_item("Hint_window", show=False)
    dpg.configure_item("video_loading_hint", show=False)
    for t in thread:
        if t is not None and t.is_alive():
            t.join()
    for t in thread_video:
        if t is not None and t.is_alive():
            t.join()
    for i in range(1, 16):
        dpg.bind_item_theme(f"{i:02d}_video_button", unpress_control_video_button_theme)
    stop_event.clear()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return 0

def play_next_preset():
    global out_file, video_play_number
    global start_record, play_over, record_over, start_play
    global loaded, v_frame
    global show_hint

    if video_play_number < video_count:
        out_file = cv2.VideoWriter(os.path.join(save_case_path, f"{video_play_number:02d}.mp4"),
                                   cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                                   (int(1280), int(720)))
    start_record = False
    play_over = False
    start_play = False
    record_over = False
    loaded = False
    show_hint = False
    v_frame = []
    stop_event.clear()

def go_next_video(sender, app_data = None, user_data = None):
    if sender == "go previous":
        play_previous()
    elif sender == "go next":
        play_next_preset()
    dpg.configure_item("alert_window", show=False)

def play_previous():
    global start_record, record_over
    global play_over, start_play
    global video_play_number
    global pause
    global case_name_set
    global demo_video
    global stop_event, thread, out_file
    global frame_now, frame_count
    global record_frame_count
    global show_hint
    global loaded, v_frame

    loaded = False
    show_hint = False
    record_frame_count = 0
    frame_now = 0
    frame_count = None
    start_record = False
    play_over = False
    start_play = False
    record_over = False
    video_play_number = max(1, video_play_number - 1)
    demo_video = None
    out_file = cv2.VideoWriter(os.path.join(save_case_path, f"{video_play_number:02d}.mp4"),
                               cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                               (int(1280), int(720)))
    try:
        pygame.mixer.music.unload()
        pygame.mixer.quit()
        pygame.mixer.init()
    except:
        pass
    dpg.configure_item("End Record", show=False)
    dpg.configure_item("REC_ICON", show=False)
    dpg.configure_item("Hint_window", show=False)
    dpg.configure_item("video_loading_hint", show=False)
    # if video_play_number < 1:
    #     dpg.configure_item("Record Previous Again", show=False)
    for t in thread:
        if t is not None:
            t.join()
    for t in thread_video:
        if t is not None:
            t.join()
    for i in range(1, 16):
        if i == video_play_number:
            dpg.bind_item_theme(f"{i:02d}_video_button", press_control_video_button_theme)
        else:
            dpg.bind_item_theme(f"{i:02d}_video_button", unpress_control_video_button_theme)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    stop_event.clear()
    return 0

def set_case_name(sender, app_data, user_data):
    global case_name, case_name_set, save_case_path, out_file
    global hidden_theme, warning_string
    global video_play_number
    if not case_name_set:
        case_name = dpg.get_value(user_data)
        save_case_path = os.path.join(save_root_path, case_name + "_" + datetime.now().strftime("%y%m%d%H%M"))
        os.makedirs(os.path.join(save_root_path, case_name + "_" + datetime.now().strftime("%y%m%d%H%M")), exist_ok=True)
        out_file = cv2.VideoWriter(os.path.join(save_case_path, f"{video_play_number:02d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (int(1280), int(720)))
        case_name_set = True
        dpg.bind_item_theme("warn_text", hidden_theme)
        dpg.configure_item(user_data, enabled = False)
        dpg.set_item_label(sender, "Change")
    else:
        case_name_set = False
        dpg.bind_item_theme("warn_text", warning_string)
        dpg.configure_item(user_data, enabled = True)
        dpg.set_item_label(sender, "Ok")
        stop_event.set()
        reset_to_start()

def run_judge_program():
    global judge_program_running
    judge_program_running = True
    #原始方法
    #subprce = subprocess.Popen(["python", "UI.py"])
    #打包需求
    subprce = subprocess.Popen(["UI.exe"])
    print("子 UI 啟動")
    subprce.wait()
    print("子 UI 已關閉")
    judge_program_running = False

def button_listener(sender, app_data = None, user_data = None):
    global pause
    global record_over
    global stop_event
    global judge_program_running
    global judge_program_status

    if sender == "Start Button":
        pause = not pause
        if pause:
            stop_event.set()
            dpg.set_item_label("Start Button", "開始")
            reset_to_start()
        else:
            stop_event.clear()
            dpg.set_item_label("Start Button", "停止")
    if sender == "End Record":
        record_over = True
    if sender == "Judge Score":
        if not judge_program_running:
            judge_program_status = threading.Thread(target=run_judge_program(), daemon=True)
            judge_program_status.start()

def on_press(key):
    global play_over, start_record, record_over, case_name_set
    try:
        # print(f"Special key pressed: {key.char}")
        if case_name_set and play_over and start_record and not record_over:
            record_over = True
    except AttributeError:
        # print(f"Special key pressed: {key}")
        if case_name_set and play_over and start_record and not record_over:
            record_over = True

def keyboard_listener():
    global keyboard_listener_obj
    keyboard_listener_obj = keyboard.Listener(on_press=on_press)
    keyboard_listener_obj.start()

def video_control_block():
    global play_over, video_play_number, start_play
    global video_path
    global start_record, record_over
    global demo_video, video_fps
    global stop_event
    global press_control_video_button_theme, unpress_control_video_button_theme
    global thread_video
    global frame_count, frame_now
    global record_frame_count
    global main_screen
    global loaded

    thread_video = []
    if video_play_number >= video_count:
        if not dpg.get_item_configuration("alert_window")["show"]:
            dpg.set_value("alert_text", value="重新或繼續錄製?")
            stop_event.set()
            reset_to_start()
            return 0
    elif video_play_number < video_count:
        try:
            if video_play_number >= 1:
                for i in range(1, 16):
                    if i == video_play_number:
                        dpg.bind_item_theme(f"{i:02d}_video_button", press_control_video_button_theme)
                    else:
                        dpg.bind_item_theme(f"{i:02d}_video_button", unpress_control_video_button_theme)
        except:
            pass
        # print(dpg.get_item_configuration("End Record")["show"])
        if not start_play and not dpg.get_item_configuration("Hint_window")["show"] and not show_hint and not dpg.get_item_configuration("alert_window")["show"]:
            update_hint_box_text()
            dpg.configure_item("Hint_window", pos=(main_screen.width // 2 - 250, main_screen.height // 2 - 150), show = True)
        if not start_play and show_hint:
            print(f"Run Video {video_play_number}")
            start_play = True

            if not os.path.exists(video_path):
                print("Demo Video Not Found.")
                return

            # **正確初始化影片**
            demo_video = cv2.VideoCapture(os.path.join(video_path, f"{video_play_number:02d}.mp4"))
            video = mp.VideoFileClip(os.path.join(video_path, f"{video_play_number:02d}.mp4"))
            audio = video.audio
            audio.write_audiofile("audio.wav", codec='pcm_s16le')
            # **確保 `MediaPlayer` 正確啟動**
            # demo_video_sound = os.path.join(video_path, f"{video_play_number:02d}.mp4")
            # 取得 FPS，確保正確播放
            # video_fps = demo_video.get(cv2.CAP_PROP_FPS)

            # **正確啟動執行緒**
            t = threading.Thread(target=play_video_loop, args=(demo_video,), daemon=True)
            t.start()
            thread_video.append(t)
            t = threading.Thread(target=play_video_sound,daemon=True)
            t.start()
            thread_video.append(t)
        elif loaded and start_play and not start_record and frame_count - frame_now <= 30:
            start_record = True
            dpg.configure_item("REC_ICON", show=True)
            record_frame_count = 0
            print("開始錄影")
        elif loaded and start_play and start_record and not record_over and  record_frame_count // 30 >= 8:
            print("record time : ", record_frame_count // 30)
            record_over = True
        elif loaded and play_over and start_record and not record_over and not dpg.get_item_configuration("End Record")["show"]:
            stop_event.set()
            for i in thread_video:
                if i is not None and i.is_alive():
                    i.join()
            dpg.configure_item("End Record", show = True)
        elif loaded and play_over and start_record and record_over and dpg.get_item_configuration("End Record")["show"] and not dpg.get_item_configuration("alert_window")["show"]:
            print("結束錄影")
            dpg.configure_item("alert_window", pos=[main_screen.width // 2 - 150, main_screen.height // 2 - 75], show=True)
            dpg.configure_item("End Record", show=False)
            dpg.configure_item("REC_ICON", show=False)
            video_play_number += 1
            if video_play_number == video_count:
                dpg.set_value("alert_text", value="所有測驗已結束!!!\n重新或繼續錄製?")

def update_hint_box_text():
    global video_play_number
    match video_play_number:
        case 0:
            dpg.set_value("hint_text", "施測者請注意:測驗即將開始，\n請注意螢幕與攝影機擺放，\n椅子擺放是否適當。\n\n提示語:先練習一次，請準備。\n \n")   #00
        case 1:
            dpg.set_value("hint_text", "測驗 1\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")   #01
        case 2:
            dpg.set_value("hint_text", "測驗 2\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")   #02
        case 3:
            dpg.set_value("hint_text", "測驗 3\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")   #03
        case 4:
            dpg.set_value("hint_text", "測驗 4\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")    #04
        case 5:
            dpg.set_value("hint_text", "測驗 5\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")    #05
        case 6:
            dpg.set_value("hint_text", "測驗 6\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")    #06
        case 7:
            dpg.set_value("hint_text", "測驗 7\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")    #07
        case 8:
            dpg.set_value("hint_text", "測驗 8\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")     #08
        case 9:
            dpg.set_value("hint_text", "測驗 9\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")     #09
        case 10:
            dpg.set_value("hint_text", "測驗 10\n\n施測者請注意:接下來是站著的動作。\n\n請看示範動作，\n示範完跟著做一樣的動作。\n \n")      #10
        case 11:
            dpg.set_value("hint_text", "測驗 11\n\n施測者請注意:請小朋友站到螢幕右邊。\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")    #11
        case 12:
            dpg.set_value("hint_text", "測驗 12\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")    #12
        case 13:
            dpg.set_value("hint_text", "測驗 13\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")    #13
        case 14:
            dpg.set_value("hint_text", "測驗 14\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")    #14
        case 15:
            dpg.set_value("hint_text", "測驗 15\n\n指導語:請看示範動作，\n示範完跟著做一樣的動作。\n \n")    #15
        case 16:
            dpg.set_value("hint_text", "指導語:接下來看畫面的示範動作，\n示範完就跟著做動作。")    #16
        case 17:
            dpg.set_value("hint_text", "指導語:接下來看畫面的示範動作，\n示範完就跟著做動作。")    #17
        case 18:
            dpg.set_value("hint_text", "指導語:接下來看畫面的示範動作，\n示範完就跟著做動作。")    #18
        case 19:
            dpg.set_value("hint_text", "標題:視覺穩定(左右轉動)\n指導語:接下來請先看我示範\n提醒：施測者示範完後再按繼續鍵。")  #19
        case 20:
            dpg.set_value("hint_text", "標題:視覺穩定(上下轉動)\n指導語:接下來請先看我示範\n提醒：施測者示範完後再按繼續鍵。")  #20
        case _:
            print(f"No action defined for video_play_number {video_play_number}")

def play_video_loop(demo_video):
    global play_over
    global MS_video_window_height, MS_video_window_width
    global second_screen
    global video_fps
    global stop_event
    global record_over, case_name_set, pause
    global frame_now
    global loaded, v_frame
    global frame_count

    # 初始化 pygame 顯示（設定為第二螢幕位置）
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{second_screen.x},{second_screen.y}"
    pygame.init()
    video_window = pygame.display.set_mode((second_screen.width, second_screen.height))
    pygame.display.set_caption("Demo Video")

    video_fps = demo_video.get(cv2.CAP_PROP_FPS)
    frame_count = demo_video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_now = 0  # 計算當前播放到第幾幀
    # 讀取整部影片進記憶體
    v_frame = []
    print("load video")
    dpg.configure_item("video_loading_hint", show=True)
    while not stop_event.is_set():
        if not dpg.is_dearpygui_running():
            break
        ret, frame = demo_video.read()
        if not ret:
            loaded = True
            break
        v_frame.append(frame)
    start_time = time.time()  # 記錄影片開始播放的時間
    print("video loaded")
    dpg.configure_item("video_loading_hint", show=False)

    while not play_over:
        if stop_event.is_set():
            print("接收終止信號 關閉影片")
            break
        if frame_now >= len(v_frame):
            # 顯示最後一貞
            frame = v_frame[0]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resize_frame = cv2.resize(rgb_frame, (second_screen.width, second_screen.height),
                                      interpolation=cv2.INTER_NEAREST)
            surface = pygame.surfarray.make_surface(np.transpose(resize_frame, (1, 0, 2)))
            video_window.blit(surface, (0, 0))
            pygame.display.update()
            play_over = True
            stop_event.set()
        else:
            frame = v_frame[frame_now]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resize_frame = cv2.resize(rgb_frame, (MS_video_window_width, MS_video_window_height),
                                      interpolation=cv2.INTER_NEAREST)
            dpg_frame = resize_frame.astype(np.float32).ravel() / 255.0

            # 用 pygame 顯示影片畫面
            resize_frame = cv2.resize(rgb_frame, (second_screen.width, second_screen.height),
                                      interpolation=cv2.INTER_NEAREST)
            surface = pygame.surfarray.make_surface(np.transpose(resize_frame, (1, 0, 2)))
            video_window.blit(surface, (0, 0))
            pygame.display.update()

            # 更新 DPG 紋理（每 2 幀更新一次）
            if frame_now % 2 == 0:
                dpg.set_value("DEMS_TEXTURE", dpg_frame)

            # FPS 控制（與影片同步）
            expected_time = start_time + ((frame_now + 1) / video_fps)
            sleep_time = max(0, expected_time - time.time())
            time.sleep(sleep_time)

            # 處理 pygame 事件（避免 Not Responding）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    play_over = True
                    stop_event.set()
                    break

            frame_now += 1
    while True:
        if play_over and record_over:
            pygame.quit()
            break
        else:
            pass

def play_video_sound():
    global stop_event, play_over
    global loaded
    """ 使用 VLC 播放音訊 """
    while True:
        if loaded:
            break
    pygame.mixer.music.load("audio.wav")
    pygame.mixer.music.play()
    while True:
        if play_over or stop_event.is_set():
            # demo_video_sound.close_player()
            try:
                pygame.mixer.music.unload()
            except:
                pass
            print("接收終止信號，關閉播放器")
            break
        time.sleep(0.1)

def record_control_block(frame):
    global start_record, record_over, out_file
    global record_frame_count

    if start_record and not record_over:
        if out_file is None or not out_file.isOpened():
            print("Error: VideoWriter is not initialized!")
            return  # 避免寫入失敗
        out_file.write(frame)
        record_frame_count += 1

    elif start_record and record_over:
        if out_file is not None and out_file.isOpened():
            out_file.release()
            out_file = None

def camera_control_block():
    global MS_video_window_height, MS_video_window_width
    global camera
    global stop_event
    global start_record

    while True:
        ret, frame = camera.read()
        if not dpg.is_dearpygui_running():
            break
        if ret:
            resize_frame = cv2.resize(frame, (MS_video_window_width, MS_video_window_height))
            resize_frame = np.flip(resize_frame, axis=2)  # 翻轉 BGR->RGB
            resize_frame = resize_frame.astype(np.float32).ravel() / 255.0
            dpg.set_value("LIVE_TEXTURE", resize_frame)
            if start_record:
                record_control_block(frame)
        else:
            print("No Camera Frame")
            break

def main_loop():
    global camera, demo_video, demo_video_sound, video_play_number
    global video_count
    global case_name_set, pause, camera_is_open
    global thread
    global listen_keyboard

    if listen_keyboard is None or not listen_keyboard.is_alive():
        listen_keyboard = threading.Thread(target=keyboard_listener, daemon=True)
        listen_keyboard.start()
    init_parameter()
    dearpygui_setup()
    start_time = time.time()
    image_count = 0
    while dpg.is_dearpygui_running():
        """Main Logic"""
        # print(case_name_set, pause)
        if case_name_set and not pause:
            video_control_block()
            if not camera_is_open:
                camera_is_open = True
                thread.append(threading.Thread(target = camera_control_block, daemon=True).start())
        # if video_play_number is not None and video_play_number < 1:
        #     dpg.configure_item("Record Previous Again", show=False)
        # else:
        #     dpg.configure_item("Record Previous Again", show=True)
        for i in range(1, 16):
            if i == video_play_number:
                dpg.bind_item_theme(f"{i:02d}_video_button", press_control_video_button_theme)
            else:
                dpg.bind_item_theme(f"{i:02d}_video_button", unpress_control_video_button_theme)
        dpg.render_dearpygui_frame()
        """Count FPS"""
        now_time = time.time()
        if now_time - start_time >= 1:
            print(f"FPS : {image_count / (now_time - start_time):.2f}")
            image_count = 0
            start_time = time.time()
        else:
            image_count += 1
    stop_event.set()
    cleanup()
    # stop_event.set()
    # for t in thread:
    #     if t is not None and t.is_alive():
    #         t.join()
    # pygame.mixer.music.stop()
    # pygame.mixer.quit()
    # camera.release()
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

def cleanup():
    global camera, out_file, thread, thread_video
    global listen_keyboard, keyboard_listener_obj
    print("clean all element and thread")

    try:
        reset_to_start()
    except:
        pass
    if keyboard_listener_obj is not None:
        keyboard_listener_obj.stop()
    if listen_keyboard is not None and listen_keyboard.is_alive():
        listen_keyboard.join()
    # 停止所有執行緒
    for t in thread:
        if t is not None and t.is_alive():
            t.join()
    for t in thread_video:
        if t is not None and t.is_alive():
            t.join()

    # 釋放相機
    if camera is not None and camera.isOpened():
        camera.release()

    # 釋放 OpenCV 視窗
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except:
        pass
    if out_file is not None and out_file.isOpened():  # ✅ 安全釋放
        out_file.release()
        out_file = None
    # 釋放音訊
    if (pygame.mixer.get_init()):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass
    # 釋放 pygame 顯示資源
    try:
        pygame.display.quit()
        pygame.quit()
    except:
        pass
    # 釋放影片寫入
    if out_file is not None:
        out_file.release()
        out_file = None

    if os.path.exists("audio.wav"):
        os.remove("audio.wav")
        print(f"audio.wav 已刪除")
    else:
        print(f"audio.wav 不存在")

    print("所有資源已成功釋放!")

if __name__ == "__main__":
    try:
        main_loop()
    finally:
        cleanup()