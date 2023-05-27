import argparse
from PIL import Image, ImageDraw, ImageFont
import time
import cv2
import numpy as np
import torch
import csv
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
from train_my import Net 
import glob
key_dict = {0:"正坐",1:"Right  head",2:"Left crooked head",3:"Right shoulder lift",4:"Left shoulder lift",5:"Bow head",6:"face upward"}
classification = {0:"正坐",1:"右歪头",2:"左歪头",3:"右抬肩",4:"左抬肩",5:"低头",6:"仰头"}

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img

class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
      
        
        return img

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
def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()
       

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def get_set_pose(data,threshold = {"angle_shoulder_neck":[65.0,100],"rate_shoulder_neck":[0.5,1.0],"slope_2_shoulder":[20,-20]}):
    
    def process(data):
        P = []
        pose_L = [['1x','1y'],['2x','2y'],['5x','5y'],['0x','0y'],['14x','14y'],['16x','16y'],['15x','15y'],['17x','17y']]
        for i in pose_L:
            P.append([data.get(i[0]),data.get(i[1])])
        return P

    P = np.array(process(data))
    vector_1 = P[1] - P[0]
    vector_2 = P[3] - P[0]
    left_shoulder = np.sqrt(vector_1.dot(vector_1))
    neck = np.sqrt(vector_2.dot(vector_2))
    dian = vector_1.dot(vector_2)
    cos_ = dian/(left_shoulder * neck)
    angle_shoulder_neck = np.arccos(cos_)*180/np.pi
    rate_shoulder_neck = neck / left_shoulder
    vector_3 = P[0] - P[2]
    slope_2_shoulder = vector_3[1]
    print(angle_shoulder_neck,rate_shoulder_neck,vector_3)
    if angle_shoulder_neck < threshold.get("angle_shoulder_neck")[0]:
        return 2
    elif angle_shoulder_neck > threshold.get("angle_shoulder_neck")[1]:
        return 1
    elif rate_shoulder_neck < threshold.get("rate_shoulder_neck")[0]:
        return 5
    elif rate_shoulder_neck > threshold.get("rate_shoulder_neck")[1]:
        return 6
    elif slope_2_shoulder > threshold.get("slope_2_shoulder")[0]:
        return 3
    elif slope_2_shoulder < threshold.get("slope_2_shoulder")[1]:
        return 4
    else:
        return 0
        
    
        
    
    
    
def run_demo(net, classification_net,image_provider,mode, height_size, cpu, track, smooth):
    """
    mode == 0 : 图片
    mode == 1 : 摄像头
    mode == 2 : 视频
    """
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    i = 0
    f = open('data.txt','w')

    def inner(img):    
        nonlocal num_keypoints
        nonlocal previous_poses
        nonlocal delay
        start = time.time()
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)
        xxx = pose.draw(img)
        # print(xxx)
        index = get_set_pose(xxx)
        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        
        end = time.time()
        fps  = 1.0/(start - end)
        imgL = cv2AddChineseText(orig_img,f"{classification.get(index)} fps:{fps:.1f}",(20, 20),(255, 0, 0), 30)
        # # cv2.putText(img, f"{key_dict.get(index)}", (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        img = cv2.addWeighted(imgL, 0.5, img, 0.5, 0)

        
        #for pose in current_poses:
            #cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                        #(pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))

        return img

    if mode == 0:
        img = inner(image_provider)
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
           
    else:
        for img in image_provider:
            
            img = inner(img)
            cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    #parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default=0, help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("./checkpoint/checkpoint_iter_370000.pth", map_location='cuda')
    classification_net = torch.load("./checkpoint/rua.pth")
    classification_net.eval()
    load_state(net, checkpoint)
    # if args.video != '':
    #     
    # else:
    #     args.track = 0
    mode = 1
    if mode == 0:
        # frame_provider = ImageReader(glob.glob("./img/*.jpg"))
        frame_provider = cv2.imread("./img/1772.jpg", cv2.IMREAD_COLOR)
    elif mode == 1:
        frame_provider = VideoReader(args.video)
    else:
        frame_provider = VideoReader("./test_data/video.mp4")
    run_demo(net, classification_net,frame_provider, mode,args.height_size, args.cpu, args.track, args.smooth)
