import time
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import pandas as pd
# ------------------------------------------------
# 0) Mediapipe 取 pose + 同幀正規化（保留時間軸：沒偵測到一樣塞佔位）
# ------------------------------------------------
class MDP:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23

    def __init__(self):
        self.mp_holistic = mp.solutions.holistic

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        pose_frames = []  # list[ list[(x,y)] or None ]; None 代表本幀沒偵測到 pose
        count = 0
        try:
            with self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as holistic:
                while True:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    res = holistic.process(rgb)

                    if not res.pose_landmarks:
                        pose_frames.append(None)  # 佔位
                        count += 1
                        continue

                    pose = res.pose_landmarks.landmark
                    # 正規化（中心=肩中點；尺度=肩-髖距）
                    ls, rs, lh = pose[self.LEFT_SHOULDER], pose[self.RIGHT_SHOULDER], pose[self.LEFT_HIP]
                    cx = (ls.x + rs.x) / 2.0
                    cy = (ls.y + rs.y) / 2.0
                    unit = ((ls.x - lh.x) ** 2 + (ls.y - lh.y) ** 2) ** 0.5
                    if unit <= 1e-8:
                        unit = 1.0

                    norm = []
                    for lm in pose:
                        nx = (lm.x - cx) / unit
                        ny = (lm.y - cy) / unit
                        norm.append((nx, ny))
                    pose_frames.append(norm)
                    count += 1
        finally:
            cap.release()
            print(f"Mediapipe processed frames: {count}")
        return pose_frames  # list of 33x2（或 None）


# ------------------------------------------------
# 1) 補值器：把 list 轉 (T,33,2) → 補短缺口 → 平滑（NaN-safe）
# ------------------------------------------------
class Interpolator:
    N_POSE = 33

    @staticmethod
    def pack_xy(pose_frames):
        """把 list 轉成 (T,33,2)，缺值填 NaN。"""
        T = len(pose_frames)
        A = np.full((T, Interpolator.N_POSE, 2), np.nan, dtype=np.float32)
        for t, frame in enumerate(pose_frames):
            if frame is None:
                continue
            for i, (x, y) in enumerate(frame[:Interpolator.N_POSE]):
                A[t, i, 0] = x
                A[t, i, 1] = y
        return A

    @staticmethod
    def _interp_1d(x, max_gap=6):
        """線性補值：只補連續 NaN 長度 ≤ max_gap 的短缺口；長缺口保 NaN。"""
        x = x.astype(np.float32)
        idx = np.arange(len(x))
        mask = ~np.isnan(x)
        if mask.sum() == 0:
            return x
        y = x.copy()
        y[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
        # 還原長缺口
        run = 0
        for i, v in enumerate(x):
            if np.isnan(v):
                run += 1
            else:
                if run > max_gap:
                    y[i - run : i] = np.nan
                run = 0
        if run > max_gap:
            y[len(x) - run :] = np.nan
        return y

    @staticmethod
    def _smooth_valid(x, win=5):
        """忽略 NaN 的滑動平均：只在分母>0處做除法。"""
        if win <= 1:
            return x
        valid = ~np.isnan(x)
        if valid.sum() < 2:
            return x
        num = np.convolve(np.where(valid, x, 0.0), np.ones(win, dtype=np.float32), mode="same")
        den = np.convolve(valid.astype(np.float32), np.ones(win, dtype=np.float32), mode="same")
        out = np.full_like(num, np.nan, dtype=np.float32)
        np.divide(num, den, out=out, where=den > 0)
        return out

    @staticmethod
    def interpolate_pose(pose_frames, max_gap=6, smooth_win=5):
        """回傳 (T,33,2) 的補值後 pose 座標（仍可能含長缺口 NaN）。"""
        A = Interpolator.pack_xy(pose_frames)
        T, N, _ = A.shape
        for j in range(N):
            for ax in (0, 1):
                v = A[:, j, ax]
                v = Interpolator._interp_1d(v, max_gap=max_gap)
                v = Interpolator._smooth_valid(v, win=smooth_win)
                A[:, j, ax] = v
        return A


# ------------------------------------------------
# 2) 通用 FSM（State / Transition / Machine）— 支援 frame/time 兩種時鐘
# ------------------------------------------------
class State:
    def __init__(self, name,
                 min_hold_frames=None, timeout_frames=None,
                 min_hold_ms=None, timeout_ms=None,
                 on_enter=None, on_exit=None):
        self.name = name
        self.min_hold_frames = min_hold_frames
        self.timeout_frames = timeout_frames
        self.min_hold_ms = min_hold_ms
        self.timeout_ms = timeout_ms
        self.on_enter = on_enter
        self.on_exit = on_exit
        self.enter_i = None  # 幀索引
        self.enter_t = None  # 時間戳（秒）

class Transition:
    def __init__(self, src, dst, cond, name=None):
        self.src = src; self.dst = dst; self.cond = cond
        self.name = name or f"{src.name}->{dst.name}"

class Machine:
    def __init__(self, states, initial, transitions, idle_name="IDLE", clock="frame"):
        self.states = {s.name: s for s in states}
        self.cur = initial
        self.idle = self.states.get(idle_name, initial)
        self.transitions = transitions
        self.clock = clock
        self.i = 0  # 全域幀計數（每 step +1）
        self._enter(initial)
    def _enter(self, s: State):
        s.enter_i = self.i
        s.enter_t = time.time()
        if s.on_enter: s.on_enter()
    def _exit(self, s: State):
        if s.on_exit: s.on_exit()
    def step(self, obs):
        self.i += 1
        now = time.time(); cur = self.cur
        if obs is None:
            return cur.name
        # timeout
        if self.clock == "frame":
            if cur.timeout_frames is not None and (self.i - cur.enter_i) >= cur.timeout_frames:
                target = self.idle
                if target != cur: self._exit(cur); self.cur = target; self._enter(self.cur)
                return self.cur.name
        else:
            if cur.timeout_ms is not None and (now - cur.enter_t) * 1000 >= cur.timeout_ms:
                target = self.idle
                if target != cur: self._exit(cur); self.cur = target; self._enter(self.cur)
                return self.cur.name
        # min hold
        if self.clock == "frame":
            if cur.min_hold_frames is not None and (self.i - cur.enter_i) < cur.min_hold_frames:
                return cur.name
        else:
            if cur.min_hold_ms is not None and (now - cur.enter_t) * 1000 < cur.min_hold_ms:
                return cur.name
        # transitions（依序檢查，只跳一次）
        for t in self.transitions:
            if t.src == cur and t.cond(obs):
                self._exit(cur); self.cur = t.dst; self._enter(self.cur); break
        return self.cur.name


# ------------------------------------------------
# 3) Action1V3：無互斥 + 以 delta 為高度度量 + 新四項分數
# ------------------------------------------------
class Action1V3:
    R_WRIST, R_HIP = 16, 24
    L_WRIST, L_HIP = 15, 23

    def __init__(self,
                 # --- 判斷門檻（高度版，不用速度） ---
                 raise_margin=0.10,  # 進入 UP 的上門檻（抬到這麼「高」才算上舉）
                 tap_low=0.04,  # 進入 TAP 的下門檻（靠近底部到這個範圍算回到大腿）
                 # --- 去抖 / 互動 ---
                 tap_lock_frames=8,  # 同側 TAP 事件的最小間隔幀數（避免底部抖動連續記 TAP）
                 hold_frames=4,  # FSM 最短停留幀數（防抖；每進入一個狀態至少停留這麼久）
                 timeout_frames=180,  # FSM 超時回復幀數（卡住太久自動回 IDLE）
                 # --- 底線（baseline）濾波 ---
                 ema_alpha=0.20  # 在 TAP 區域更新 baseline 的 EMA 係數（越小越穩）
                 ):
        # ===== 1) 參數保存 =====
        # 門檻（高度版雙門檻）
        self.raise_margin = float(raise_margin)  # delta >= raise_margin → 進 UP
        self.tap_low = float(tap_low)  # delta <= tap_low     → 進 TAP

        # 去抖/時間參數
        self.tap_lock_frames = int(tap_lock_frames)
        self.ema_alpha = float(ema_alpha)

        # ===== 2) 幀計數 =====
        self.frame_i = 0  # 目前處理到第幾幀（Machine(clock='frame') 會用得到）

        # ===== 3) 每側狀態記憶（給 _obs / step 使用）=====
        # 上一幀手腕 y（用來初始化/追蹤；高度版不再用速度，但保留以便除錯）
        self.prev_wy_R = None
        self.prev_wy_L = None

        # baseline（底部休息位）的手腕 y；只在 TAP 區域用 EMA 更新
        self.base_wy_R = None
        self.base_wy_L = None

        # ----- 同手 TAP→TAP 區間追蹤（分數二 / 分數四）-----
        self.prev_tap_R = False
        self.prev_tap_L = False
        self.last_tap_frame_R = None  # 上一次右手 TAP 的幀
        self.last_tap_frame_L = None  # 上一次左手 TAP 的幀
        self.saw_up_since_last_tap_R = False  # 自上一個 TAP 之後是否曾進入過 UP（單循環約束）
        self.saw_up_since_last_tap_L = False
        self.peak_delta_since_last_tap_R = None  # 這個右手 TAP→下一個右手 TAP 期間的最大 delta
        self.peak_delta_since_last_tap_L = None  # 左手同理

        self.same_side_tap_intervals = []  # [右TAP→右TAP, 左TAP→左TAP, ...]（單位：幀）
        self.peak_delta_between_taps = []  # 對應每個同手區間的 peak delta

        # ----- 跨手 TAP 間隔追蹤（分數三）-----
        self.tap_seq_events = []  # 全域 TAP 序列（只在 UP→TAP 時記 'R'/'L'）
        self.tap_event_frames = []  # 對應 TAP 發生幀（與 tap_seq_events 對齊）
        self.cross_tap_intervals = []  # 相鄰 TAP（必不同側）之間的幀距

        # ----- 視覺化/除錯用 -----
        self.delta_trace_R = []  # 每幀右手 delta（可含 None）
        self.delta_trace_L = []  # 每幀左手 delta
        self.baseline_trace_R = []  # 每幀右手 baseline_wy（可含 None）
        self.baseline_trace_L = []  # 每幀左手 baseline_wy

        # ===== 4) 建立左右兩套 FSM（不做互斥限制）=====
        # 狀態定義：IDLE / TAP / UP
        idleR = State("IDLE", min_hold_frames=hold_frames, timeout_frames=timeout_frames)
        tapR = State("TAP", min_hold_frames=hold_frames, timeout_frames=timeout_frames)
        upR = State("UP", min_hold_frames=hold_frames, timeout_frames=timeout_frames)

        # 右手轉移：只看高度門檻（_to_up / _to_tap）
        self.fsm_R = Machine(
            [idleR, tapR, upR],
            initial=idleR,
            clock="frame",
            transitions=[
                Transition(idleR, upR, lambda o: self._to_up(o)),  # 底部 → 抬升
                Transition(idleR, tapR, lambda o: self._to_tap(o)),  # 尚未抬起但已在底部
                Transition(tapR, upR, lambda o: self._to_up(o)),
                Transition(upR, tapR, lambda o: self._to_tap(o)),  # 抬完回到底部
            ]
        )

        idleL = State("IDLE", min_hold_frames=hold_frames, timeout_frames=timeout_frames)
        tapL = State("TAP", min_hold_frames=hold_frames, timeout_frames=timeout_frames)
        upL = State("UP", min_hold_frames=hold_frames, timeout_frames=timeout_frames)

        # 左手轉移：同右手
        self.fsm_L = Machine(
            [idleL, tapL, upL],
            initial=idleL,
            clock="frame",
            transitions=[
                Transition(idleL, upL, lambda o: self._to_up(o)),
                Transition(idleL, tapL, lambda o: self._to_tap(o)),
                Transition(tapL, upL, lambda o: self._to_up(o)),
                Transition(upL, tapL, lambda o: self._to_tap(o)),
            ]
        )

        # ===== 5) 其他：供外部取用的細節容器 =====
        self.details = {}  # 你在 get_scores()/summary() 裡可以把中間結果塞進來做除錯輸出

    # --- 觀測 ---
    def _obs(self, pose_xy, w_idx, prev_wy, base_wy):
        """
        產生一側的觀測值（不靠 HIP）：
          - wy: 當幀手腕 y
          - delta = base_wy - wy
          - tap：delta <= tap_low
          - 若 tap：用 EMA 更新 base_wy（底部自校準）
        """
        wy = pose_xy[w_idx, 1]
        if np.isnan(wy):
            return None, prev_wy, base_wy

        # 初始化 baseline（第一幀）
        if base_wy is None:
            base_wy = wy

        delta = float(base_wy - wy)  # 抬高手度（>0 代表抬高）
        tap = bool(delta <= self.tap_low)  # 高度回到底部附近

        # 只在 TAP 區用 EMA 慢慢貼合底部
        if tap:
            base_wy = self.ema_alpha * wy + (1.0 - self.ema_alpha) * base_wy

        # 高度版不需要速度，保留 v=None
        o = {"wrist_y": wy, "v": None, "tap": tap, "delta": delta}
        return o, wy, base_wy

    # --- 規則（不檢查互斥）---
    def _to_up(self, o):
        return (o is not None) and (o["delta"] is not None) and (o["delta"] >= self.raise_margin)

    def _to_tap(self, o):
        return (o is not None) and (o["delta"] is not None) and (o["delta"] <= self.tap_low)

    # --- 推進 ---
    def step(self, pose_xy_frame):
        self.frame_i += 1
        oR, self.prev_wy_R, self.base_wy_R = self._obs(pose_xy_frame, self.R_WRIST, self.prev_wy_R, self.base_wy_R)
        oL, self.prev_wy_L, self.base_wy_L = self._obs(pose_xy_frame, self.L_WRIST, self.prev_wy_L, self.base_wy_L)
        # 紀錄delta
        self.delta_trace_L.append(None if (oL is None) else oL["delta"])
        self.delta_trace_R.append(None if (oR is None) else oR["delta"])

        # 先前狀態（若需）
        # pR, pL = self.fsm_R.cur.name, self.fsm_L.cur.name
        # 推進 FSM
        sR, sL = self.fsm_R.step(oR), self.fsm_L.step(oL)

        # 在「同手兩 TAP 之間」更新 peak delta（delta 最大值）
        if oR is not None and oR["delta"] is not None and self.last_tap_frame_R is not None:
            self.peak_delta_since_last_tap_R = oR["delta"] if self.peak_delta_since_last_tap_R is None \
                else max(self.peak_delta_since_last_tap_R, oR["delta"])
        if oL is not None and oL["delta"] is not None and self.last_tap_frame_L is not None:
            self.peak_delta_since_last_tap_L = oL["delta"] if self.peak_delta_since_last_tap_L is None \
                else max(self.peak_delta_since_last_tap_L, oL["delta"])

        # 有沒有在區間內進過 UP（用於同手 TAP→TAP 篩選）
        if sR == "UP":
            self.saw_up_since_last_tap_R = True
        if sL == "UP":
            self.saw_up_since_last_tap_L = True

        # TAP 上升沿偵測（右）
        if oR is not None and oR["tap"] and not self.prev_tap_R:
            if self.last_tap_frame_R is not None and (self.frame_i - self.last_tap_frame_R) < self.tap_lock_frames:
                self.prev_tap_R = True
                # 只「不記錄」這次 TAP；繼續往下讓左手照常判斷
            else:
                if self.tap_seq_events:
                    last_side = self.tap_seq_events[-1]
                    last_frame = self.tap_event_frames[-1]
                    if last_side != 'R':
                        self.cross_tap_intervals.append(self.frame_i - last_frame)
                self.tap_seq_events.append('R');
                self.tap_event_frames.append(self.frame_i)

                if self.last_tap_frame_R is not None and self.saw_up_since_last_tap_R:
                    self.same_side_tap_intervals.append(self.frame_i - self.last_tap_frame_R)
                    if self.peak_delta_since_last_tap_R is not None:
                        self.peak_delta_between_taps.append(self.peak_delta_since_last_tap_R)

                self.last_tap_frame_R = self.frame_i
                self.saw_up_since_last_tap_R = False
                self.peak_delta_since_last_tap_R = oR["delta"] if (oR is not None) else None
        self.prev_tap_R = (oR is not None and oR["tap"])

        # TAP 上升沿偵測（左）
        if oL is not None and oL["tap"] and not self.prev_tap_L:
            if self.last_tap_frame_L is not None and (self.frame_i - self.last_tap_frame_L) < self.tap_lock_frames:
                self.prev_tap_L = True  # 更新邊緣旗標，避免連續觸發
            else:
                if self.tap_seq_events:
                    last_side = self.tap_seq_events[-1]
                    last_frame = self.tap_event_frames[-1]
                    if last_side != 'L':
                        self.cross_tap_intervals.append(self.frame_i - last_frame)
                self.tap_seq_events.append('L'); self.tap_event_frames.append(self.frame_i)
                if self.last_tap_frame_L is not None and self.saw_up_since_last_tap_L:
                    self.same_side_tap_intervals.append(self.frame_i - self.last_tap_frame_L)
                    if self.peak_delta_since_last_tap_L is not None:
                        self.peak_delta_between_taps.append(self.peak_delta_since_last_tap_L)
                self.last_tap_frame_L = self.frame_i
                self.saw_up_since_last_tap_L = False
                self.peak_delta_since_last_tap_L = oL["delta"] if (oL is not None) else None
        self.prev_tap_L = (oL is not None and oL["tap"])

        return sR, sL

    def summary(self):
        return {
            "tap_seq": self.tap_seq_events,
            "same_side_intervals": self.same_side_tap_intervals,
            "cross_side_intervals": self.cross_tap_intervals,
            "peak_delta_between_taps": self.peak_delta_between_taps,
        }

    def _cv_to_score(self, values):
        vals = np.array(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size <= 1:
            return 10
        mean = np.mean(vals)
        std = np.std(vals)
        denom = max(1e-6, abs(mean))  # 避免接近 0
        cv = std / denom
        return float(max(0.0, min(1.0, 1.0 - cv)) * 100.0)

    def get_scores(self):
        # S1：交錯性
        if len(self.tap_seq_events) < 2:
            s1 = 10
        else:
            N = len(self.tap_seq_events)
            run = 1
            longest_run = 1
            for i in range(1, N):
                if self.tap_seq_events[i] != self.tap_seq_events[i - 1]:
                    run += 1
                    if run > longest_run:
                        longest_run = run
                else:
                    run = 1
            longest_ratio = longest_run / N
            s1 = 100.0 * longest_ratio
        # S2：同手 TAP→(含 UP)→TAP 間距一致性
        s2 = self._cv_to_score(self.same_side_tap_intervals)
        # S3：跨手 TAP→對側 TAP 間距一致性
        s3 = self._cv_to_score(self.cross_tap_intervals)
        # S4：兩 TAP 間 peak delta 一致性
        s4 = self._cv_to_score(self.peak_delta_between_taps)
        return {
            "S1": round(s1, 1),
            "S2": round(s2, 1),
            "S3": round(s3, 1),
            "S4": round(s4, 1),
            "details": self.summary(),
        }


# ------------------------------------------------
# 4) RunnerV3：讀影片 → Mediapipe → 補值 → Action1V3 → 回傳分數
# ------------------------------------------------
class RunnerV3:
    def __init__(self, video_path, plot_delta = False):
        self.video_path = video_path
        self.plot_delta = plot_delta

    def draw_delta(self, act):
        plt.figure()
        plt.subplot(2, 1, 1);
        plt.title("Left delta")
        plt.plot(act.delta_trace_L)
        plt.xlim(0, len(act.delta_trace_L))

        plt.subplot(2, 1, 2);
        plt.title("Right delta")
        plt.plot(act.delta_trace_R)
        plt.xlim(0, len(act.delta_trace_R))
        plt.show()

    def run(self):
        mdp = MDP(); raw_pose = mdp.process_video(self.video_path)
        pose_arr = Interpolator.interpolate_pose(raw_pose, max_gap=6, smooth_win=5)
        act = Action1V3()
        for t in range(pose_arr.shape[0]):
            act.step(pose_arr[t])

        scores = act.get_scores()
        print("Scores(V3):", scores)
        if self.plot_delta:
            self.draw_delta(act)
        return scores


# ------------------------------------------------
# 5) 執行（把路徑換成你的影片路徑）
# ------------------------------------------------
if __name__ == "__main__":
    root_path = r"C:\Bilateral Coordination Record Video"
    rows = []
    for root, dirs, files in os.walk(root_path):
        for name in files:
            if name.endswith("01.mp4"):
                video_path = os.path.join(root, name)
                try:
                    print(video_path)
                    # 跑 V3：預期回傳 dict，含四個分數
                    scores = RunnerV3(video_path, plot_delta = False).run()
                    # 盡量相容不同 key 取值（如果你的 RunnerV3 用其他命名可在這裡改）
                    s1 = scores.get("S1")
                    s2 = scores.get("S2")
                    s3 = scores.get("S3")
                    s4 = scores.get("S4")

                    rows.append({
                        "filename": video_path.split('\\')[-2],
                        "S1": s1,
                        "S2": s2,
                        "S3": s3,
                        "S4": s4,
                    })
                except Exception as e:
                    # 若單支影片出錯，保留錯誤訊息方便排查
                    rows.append({
                        "filename": name,
                        "S1": None, "S2": None, "S3": None, "S4": None,
                        "error": str(e),
                    })
    out_xlsx  = os.path.join(root_path, "scores.xlsx")
    df_new = pd.DataFrame(rows)
    df_new.to_excel(out_xlsx, index=False)
    print(f"寫入完成：{out_xlsx}（{len(df_new)} 筆）")