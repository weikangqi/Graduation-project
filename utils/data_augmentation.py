
import cv2
 

cap = cv2.VideoCapture("halfsize3.mp4")  #读取视频文件

i = 1
while(True):
    ret, frame = cap.read()
  
    
    if ret:
        #frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        cv2.imshow("frame", frame)
        cv2.imwrite("./img/{}.jpg".format(i),frame)
        i = i + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
