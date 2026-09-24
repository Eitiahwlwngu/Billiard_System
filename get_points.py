import cv2
import os
import numpy as np

# --- 配置区 ---
# 请确保这是你真实的视频路径
video_path = r"E:\python_code\Billiard_System\backend\videos\test1.mp4"

if not os.path.exists(video_path):
    print(f"❌ 错误：找不到视频文件 {video_path}")
    exit()

points = []
backup_img = None


def click_event(event, x, y, flags, params):
    global points, img, backup_img

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 6:
            points.append((x, y))
            print(f"📍 记录坐标 {len(points)}/6: ({x}, {y})")

            # 绘制点和编号
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(img, str(len(points)), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 如果凑齐6个点，画出多边形轮廓以便核对
            if len(points) == 6:
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                print("\n✅ 6个袋口坐标采集完毕！请将下方列表复制到 scoring.py 中：")
                print("=" * 50)
                print(f"self.pockets = {points}")
                print("=" * 50)
                print("按任意键退出...")

            cv2.imshow("Calibration", img)


print("💡 提示：左键点击袋口中心。如果不小心点错，按 'z' 键可以撤销。")
cap = cv2.VideoCapture(video_path)
ret, original_img = cap.read()
if not ret:
    print("❌ 无法读取视频帧")
    exit()

img = original_img.copy()
cv2.imshow("Calibration", img)
cv2.setMouseCallback("Calibration", click_event)

# 监听按键，支持 Z 键撤销
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('z') or key == ord('Z'):
        if len(points) > 0:
            points.pop()
            print("↩️ 已撤销上一个点")
            # 恢复画面重画
            img = original_img.copy()
            for i, p in enumerate(points):
                cv2.circle(img, p, 6, (0, 255, 0), -1)
                cv2.putText(img, str(i + 1), (p[0] + 10, p[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Calibration", img)
    elif key != 255:  # 按了其他任意键且采集满了6个点
        if len(points) == 6:
            break

cv2.destroyAllWindows()
cap.release()