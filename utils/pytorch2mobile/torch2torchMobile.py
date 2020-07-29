import torch
from torch import quantization, jit
from torch import load, save
import torchvision


def load_model(nn, file_name):
    nn.load_state_dict(torch.load(file_name))
    return

def save_model(nn, file_name):
    torch.save(nn.state_dict(), file_name + '.pt')
    return

def save_model_to_mobile(model=None, file_path="./model.pt", device="cpu"):
    if model is None:
        return
    
    model.eval()
    example = torch.rand(1, 3, 28, 28).to(device)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(file_path)
    
    return

def save_model_to_mobile_RelationNet(model=None, file_path=["./encoder.pt","./rn.pt"], device="cpu"):
    if model is None:
        return
    
    assert len(model) == 2
    
    encoder = model[0]
    rn = model[1]

    encoder.eval()
    example = torch.rand(10, 3, 28, 28).to(device)
    traced_script_module = torch.jit.trace(encoder, example)
    traced_script_module.save(file_path[0])
    
    rn.eval()
    example = torch.rand(10, 128, 5, 5).to(device)
    traced_script_module = torch.jit.trace(rn, example)
    traced_script_module.save(file_path[1])

    print("save success")
    return

# from AcousDigits_few_shot_test.AcousDigits_test_one_shot import CNNEncoder, RelationNetwork

# if __name__ == "__main__":

    # encoder = CNNEncoder()
    # rn = RelationNetwork(64, 8)
    # load_model(encoder, "C:/Users/KyoDante/Desktop/AcouDigits/models/AcousDigits_feature_encoder_10way_5shot_trainwith_HSC_28_DQL_28_ZM_28_HYT_28_YQ_28.pkl")
    # load_model(rn, "C:/Users/KyoDante/Desktop/AcouDigits/models/AcousDigits_relation_network_10way_5shot_trainwith_HSC_28_DQL_28_ZM_28_HYT_28_YQ_28.pkl")
    # save_model_to_mobile(encoder,'./model_encoder.pt',device='cpu')
    # save_model_to_mobile(rn,'./model_rn.pt',device='cpu')
    