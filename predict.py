from train_my import Net 
import torch

if __name__ == "__main__":
    # classification_net = Net()
    # checkpoint2 = torch.load("./checkpoint/classification.pth")

    # classification_net.load_state_dict(checkpoint2,False)
    
    classification_net = torch.load("./checkpoint/rua.pth")
    classification_net.eval()
    
    # with torch.no_grad():
    inp  = torch.tensor([],dtype=torch.float)
    
    inp = inp.unsqueeze(0)
    
    
    
    
    print(inp.shape)
    out = classification_net(inp)
    
    index = torch.argmax(out, axis=1)
    print(index)
     