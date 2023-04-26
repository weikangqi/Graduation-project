# import cv2
# import numpy as np
# # point = [369, 315, 230, 311, 523, 307, 290, 262, 253, 236, 305, 217, 369, 157, 0, 0]
# # point[::2]
# # print(point[::2])
# # print(point[1::2])
# # L = [[x,y]for x,y in zip(point[::2],point[1::2])]

# # print(L)

# # img = np.ones((1000,1000),dtype = np.uint8)
# # for i in range(len(L)):
# #     img = cv2.circle(img, L[i], 3, [255,0,255], -1)
# #     cv2.imshow("img",img)
# #     cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # img = np.ones((1000,1000),dtype = np.uint8)
# # # img = cv2.circle(img, (int(100), int(100)), 3, [255,0,255], -1)
# # for i in range(len(point)//2):
# #     img = cv2.circle(img, (point[i*2], point[i*2+1]), 3, [255,0,255], -1)
# #     cv2.imshow("img",img)
# #     cv2.waitKey(0)
# # cv2.destroyAllWindows()
# d = {'1x': 339, '1y': 333, '2x': 223, '2y': 326, '5x': 463, '5y': 341, '0x': 283, '0y': 258, '14x': 256, '14y': 232, '15x': 309, '15y': 225, '17x': 361, '17y': 198, '16x': 0, '16y': 0}
# pose_L = [['1x','1y'],['2x','2y'],['5x','5y'],['0x','0y'],['14x','14y'],['16x','16y'],['15x','15y'],['17x','17y']]
# P = []
# for i in pose_L:
#     P.append([d.get(i[0]),d.get(i[1])])
# P = np.array(P)

# vector_1 = P[1] - P[0]
# vector_2 = P[3] - P[0]
# l_1 = np.sqrt(vector_1.dot(vector_1))
# l_2 = np.sqrt(vector_2.dot(vector_2))
# dian = vector_1.dot(vector_2)
# cos_ = dian/(l_1 * l_2)
# angle_hu = np.arccos(cos_)
# angle_d = angle_hu*180/np.pi
# print(angle_d)
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
cap = cv2.VideoCapture(0)
# font=cv2.FONT_ITALIC
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
while(1):
    ret,frame = cap.read()
    # 展示图片
    frame=cv2AddChineseText(frame,"劳资最帅", (123, 123),(0, 255, 0), 30)
    cv2.imshow('capture',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#释放对象和销毁窗口
cap.release()
cv2.destroyAllWindows()