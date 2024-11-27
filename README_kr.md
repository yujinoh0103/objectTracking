## 완료 여부
### **객체 탐지 알고리즘**  
| 알고리즘 | 설명 | 구현 여부 |
|----------|------|-----------|
| **YOLO (You Only Look Once)** | 고속·고정밀 객체 탐지 모델. 실시간 탐지에 적합. | ✅ |
| **Faster R-CNN** | 높은 정확도의 객체 탐지 모델. 복잡한 객체에 적합. 영상이 아닌 이미지에서 객체를 탐지하기 때문에 프레임별로 나누는 작업이 필요하다.| ✅ |
| **SSD (Single Shot MultiBox Detector)** | 실시간 탐지에 적합한 경량 모델. | ✅ |
| **EfficientDet** | 정확도와 속도 균형이 뛰어난 모델. |  |
| **RetinaNet** | Focal Loss를 사용해 작은 객체 탐지에 강점. |  |

---

### **객체 추적 알고리즘**  
| 알고리즘 | 설명 | 구현 여부 |
|----------|------|-----------|
| **SORT (Simple Online and Realtime Tracking)** | 간단하고 효율적인 실시간 추적 알고리즘. |  |
| **DeepSORT** | SORT의 확장판. 딥러닝을 활용한 고급 추적. |  |
| **ByteTrack** | 객체 검출의 신뢰도를 높여 추적 성능 향상. |  |
| **Kalman Filter** | 상태 예측 및 추정을 사용하는 고전적 추적 알고리즘. |  |
| **IOU Tracker** | 바운딩 박스 간의 IOU(Intersection Over Union)로 추적. |  |