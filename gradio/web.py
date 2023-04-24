import gradio as gr
import os
# import demo

def video_identity(video):
    print(video)
    return video


demo1 = gr.Interface(video_identity, 
                    gr.Video(), 
                    "playable_video", 
                    title = "视频",
                    cache_examples=False)
demo2 = gr.Interface(video_identity, 
                    gr.Image(), 
                    "playable_video", 
                    title = "图片",
                    cache_examples=False)

demo3 = gr.Interface(video_identity, 
                    gr.Button(), 
                    "playable_video", 
                    title = "图片",
                    cache_examples=False)


demo = gr.TabbedInterface([demo1, demo2,demo3],title="合工大 韦康琦 毕业设计 基于计算机视觉的坐姿检测系统")

if __name__ == "__main__":
    demo.launch()
    
