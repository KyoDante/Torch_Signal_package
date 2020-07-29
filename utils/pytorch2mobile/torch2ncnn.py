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

from AcousDigits_few_shot_test.AcousDigits_test_one_shot import CNNEncoder, RelationNetwork

if __name__ == "__main__":

    encoder = CNNEncoder()
    rn = RelationNetwork(64, 8)
    load_model(encoder, "C:/Users/KyoDante/Desktop/AcouDigits/models/AcousDigits_feature_encoder_10way_5shot_trainwith_HSC_28_DQL_28_ZM_28_HYT_28_YQ_28.pkl")
    load_model(rn, "C:/Users/KyoDante/Desktop/AcouDigits/models/AcousDigits_relation_network_10way_5shot_trainwith_HSC_28_DQL_28_ZM_28_HYT_28_YQ_28.pkl")
    save_model_to_mobile_RelationNet([encoder,rn],)
    
    print("load success")
    print(rn)
    t1 = torch.rand([1,3,28,28])
    e1 = encoder(t1)
    
    t2 = torch.rand([10,3,28,28])
    e2 = encoder(t2)

    e1_ext = e1.repeat(10, 1, 1, 1)
    print(e1_ext.shape)

    relation_pair = torch.cat((e2, e1_ext),dim=1)
    print(relation_pair.shape)

    result = rn(relation_pair)
    print(result.shape)

    result_s = torch.squeeze(result, dim=1)
    _, where = torch.max(result_s, dim=0)
    print(where)

    # torch => onnx
    torch.onnx._export(encoder,t1,"encoder.onnx", export_params=True)
    torch.onnx._export(rn,relation_pair,"rn.onnx", export_params=True)
    