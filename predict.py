from train_my import Net 
import torch

if __name__ == "__main__":
    # classification_net = Net()
    # checkpoint2 = torch.load("./checkpoint/classification.pth")

    # classification_net.load_state_dict(checkpoint2,False)
    
    classification_net = torch.load("./checkpoint/rua.pth")
    classification_net.eval()
    
    with torch.no_grad():
        inp  = torch.tensor([522,450,349,441,690,446,585,294,547,247,471,247,623,260,648,277],dtype=torch.float)
        print(inp.shape)
        inp = inp.unsqueeze(0)
        print(inp.shape)

        out = classification_net(inp)
        print(out)
        index = torch.argmax(out, axis=1)
        print(index)