import gradio as gr
import os
import cv2
import numpy as np
import pandas as pd
import time
def video_identity(video):
    
    print(video)
    return video
inputs = gr.inputs.Image()

def to_black(image):
   
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return output


classification = {0:"正坐",1:"右歪头",2:"左歪头",3:"右抬肩",4:"左抬肩",5:"低头",6:"仰头"}



mytab4 = gr.Blocks()
def f4(t):
    f =open("data.txt",'r')
    x = int(f.readline())
    simple = pd.DataFrame(
        {
            "坐姿": ["正坐", "右歪头", "左歪头", "右抬肩", "左抬肩", "低头", "仰头"],
            "百分比": [28, 55, 43, 91, 81, 53,  x],
        }
    )
    f.close()
    print(t)
    return simple
with mytab4 as m4:
    gr.Markdown("输入要查询的日期 \n格式 : yyyy-mm-dd")
    te = gr.Textbox()
    btn = gr.Button(value="show")
     
    station = gr.BarPlot(x="坐姿", y="百分比",vertical  =False,title="坐姿统计分析",y_title="百分比%")
    btn.click(f4,inputs=te,outputs=station)

mytab3 = gr.Blocks()
def f3():
    os.system("python demo.py")
with mytab3 as m3:
    gr.Markdown("打开前置摄像头")
   
    btn = gr.Button(value="打开")
    
    btn.click(f3,inputs=None,outputs=station)

    
function1 = gr.Interface(
                  video_identity,
                inputs="image", outputs="image",
                     
                    title = "视频",
                    cache_examples=False)
function2 = gr.Interface(fn = to_black, 
                   
                    
                    inputs="image", outputs="image",
                    cache_examples=False)



function5 = gr.Interface(video_identity, 
                    gr.Button(), 
                    "playable_video", 
                    title = "控制面板",
                    cache_examples=False)
def calculator( operation):
        print(operation)
        return operation + ["eee"]

m5 = gr.Interface(
    calculator,
    gr.CheckboxGroup(["微信", "邮件", "短信"], label="选择发送工具"),
    outputs="text",
    live=True,
) 
    
    


demo2 = gr.TabbedInterface([function1, function2,m3,m4,m5],["视频","图片","摄像头","数据分析","控制面板"],title="合工大 韦康琦 毕业设计 基于计算机视觉的坐姿检测系统")

if __name__ == "__main__":
    #f3()
    demo2.launch()
    
