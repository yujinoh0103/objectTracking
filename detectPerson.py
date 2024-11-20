import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # YOLOv8 모델 (small version)

# 비디오 파일 열기
video_path = 'samplevideo.mov'  # 비디오 파일 경로
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 비디오 끝에 도달하면 종료

    # YOLOv8을 사용하여 객체 감지
    results = model(frame)

    # 결과에서 사람 감지 (person 클래스는 COCO 데이터셋의 클래스 ID가 0)
    # results.xywh[0]는 감지된 객체에 대한 정보입니다. 
    # [0] = 사람 클래스(사람을 감지하려면 0번째 클래스를 사용)
    for result in results.xywh[0]:  # 모든 감지된 객체에 대해 반복
        if result[5] == 0:  # 사람이 감지되었을 때 (COCO에서 사람은 클래스 0)
            x1, y1, x2, y2 = map(int, result[:4])  # 바운딩 박스 좌표
            confidence = result[4]  # 신뢰도 (confidence)

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 파란색 바운딩 박스
            cv2.putText(frame, f"Person: {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 결과 비디오 화면에 표시
    cv2.imshow('YOLOv8 - Object Detection', frame)

    # 'q'를 눌러서 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 처리 종료
cap.release()
cv2.destroyAllWindows()
