
import cv2
 
#cap = cv2.VideoCapture(0)  #读取摄像头
cap = cv2.VideoCapture("3.mp4")  #读取视频文件
fps = 30
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2))
videoWriter = cv2.VideoWriter('halfsize3.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
print(size)
while(True):
    ret, frame = cap.read()
    h,w = frame.shape[:2]
    if ret:
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        videoWriter.write(frame)
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
videoWriter.release()
