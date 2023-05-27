# from train_my import Net 
# import torch

# if __name__ == "__main__":
#     # classification_net = Net()
#     # checkpoint2 = torch.load("./checkpoint/classification.pth")

#     # classification_net.load_state_dict(checkpoint2,False)
    
#     classification_net = torch.load("./checkpoint/rua.pth")
#     classification_net.eval()
    
#     # with torch.no_grad():
#     inp  = torch.tensor([],dtype=torch.float)
    
#     inp = inp.unsqueeze(0)
    
    
    
    
#     print(inp.shape)
#     out = classification_net(inp)
    
#     index = torch.argmax(out, axis=1)
#     print(index)
import gradio as gr
import pandas as pd

simple = pd.DataFrame(
    {
        "item": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        "inventory": [28, 55, 43, 91, 81, 53, 19, 87, 52],
    }
)

css = (
    "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"
)

with gr.Blocks() as demo:
    gr.BarPlot(value=simple, x="item", y="inventory", title="Simple Bar Plot").style(
        container=False,
    )

if __name__ == "__main__":
    demo.launch()