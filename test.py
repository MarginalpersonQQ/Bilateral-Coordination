import cv2
import mediapipe as mp

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 開啟影片
video_path = r"C:\Users\fangt\Desktop\YU\python_ui\BLC_judge\Bilateral-Coordination\video_input\demo_video\01.MOV"  # 修改為你的路徑
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR 轉 RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 偵測
    results = hands.process(image_rgb)

    # 畫圖
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # 取得左右手資訊（可選）
            label = hand_info.classification[0].label  # 'Left' or 'Right'
            score = hand_info.classification[0].score
            print(f"偵測到：{label} hand（信心值：{score:.2f}）")

            # 在原圖上畫出 landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 顯示結果
    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
