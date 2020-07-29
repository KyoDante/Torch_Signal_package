import torch
from torch import nn, autograd
from torch.functional import F

import sys
sys.path.append("C:/Users/KyoDante/Desktop/Torch_Signal_package/")


def grad_cam(model, img):
    """Generates a batch of grad-cams

    Args:
        model(): A cnn model
        img(tensor): a normalized batch of images

    Returns:
        grad_cams(tensor): a batch of grad-cams

    """
    
    grad_preds = model(img) # batch * n_classes

    # obtain the logit value of the predicted class for each image
    idx = grad_preds.argmax(axis=1)
    grad_preds_mine = grad_preds[list(range(grad_preds.shape[0])), idx]

    import numpy as np
    # original implementation
    # idx = torch.from_numpy(np.argmax(grad_preds.cpu().data.numpy(), axis=-1)).cuda()
    # grad_preds = torch.stack([a[i]  for a, i in zip(grad_preds, idx)])

    grad_cams = []    
    
    for i, grad_pred in enumerate(grad_preds_mine):

        #backprop for one image classification
        # model.features.zero_grad()
        # model.classifier.zero_grad()
        model.zero_grad()

        grad_pred.backward(retain_graph=True)

        #Obtain the output of the last convolutional layer 自定义
        conv_output = model.last_conv.cpu().data.numpy()[i]

        #Obtain the gradients for the last convolutional layer 自定义
        gradients = model.gradient[-1].cpu().data.numpy()[i]

        #pool gradients across channels
        weights = np.mean(gradients, axis = (1,2))

        grad_cam = np.zeros(conv_output.shape[1:], dtype=np.float32)

        #Weight each channel in conv_output
        for i, weight in enumerate(weights):
            grad_cam += weight * conv_output[i, :, :]

        # normalize the grad-cam
        import cv2
        # relu
        grad_cam = np.maximum(grad_cam, 0)

        # 缩放到原本的样子。
        grad_cam = cv2.resize(grad_cam, (28, 28)) # 根据原始图片大小修改(28, 28)
        grad_cam = grad_cam - np.min(grad_cam)
        grad_cam = grad_cam / np.max(grad_cam)
        grad_cam = torch.Tensor(grad_cam)

        grad_cams.append(grad_cam)

        import cv2
        import numpy as np
        grad_cam = np.uint8(255 * grad_cam)
        heatmap = cv2.applyColorMap(cv2.resize(grad_cam,(28, 28)), cv2.COLORMAP_JET)
        cv2.imwrite("hey1.jpg",heatmap)


    grad_cams = torch.stack(grad_cams).unsqueeze(1).cuda()

    return grad_cams


def guided_bp(model, img):
    """Generates a batch of guided-backprops
    
    Args:
        model(): A cnn model
        img_arr (Tensor): a normalized batch of images
        
    Returns: 
        guided_bps (Tensor): A batch of guided_backprops    
    
    """
    img_arr = autograd.Variable(img, requires_grad=True)

    gbp_preds = model(img_arr, guided=True)

    idx = gbp_preds.argmax(axis=1)
    gbp_preds_mine = gbp_preds[list(range(gbp_preds.shape[0])), idx]

    guided_bps = []

    for i, gbp_pred in enumerate(gbp_preds_mine):
        #backprop for one image classification
        model.classifier.zero_grad()
        model.features.zero_grad()
        gbp_pred.backward(retain_graph=True)

        #obtain the gradient w.r.t to the image
        guided_bp = img_arr.grad[i]
        guided_bps.append(guided_bp)
    
    guided_bps = torch.stack(guided_bps).cuda()

    return guided_bps


def grad_cam_pp(model, img):

    b, c, h, w = img.size()

    grad_cam_pp_preds = model(img)

    idx = grad_cam_pp_preds.argmax(axis=1)

    grad_cam_pp_preds_mine = grad_cam_pp_preds[list(range(grad_cam_pp_preds.shape[0])), idx]

    saliency_maps = []

    for i, grad_cam_pp_pred in enumerate(grad_cam_pp_preds_mine):
        #backprop for one image classification
        model.features.zero_grad()
        model.classifier.zero_grad()

        grad_cam_pp_pred.backward(retain_graph=True)

        #Obtain the output of the last convolutional layer 自定义
        conv_output = model.last_conv.cpu().data[i]

        #Obtain the gradients for the last convolutional layer 自定义
        gradients = model.gradient[-1].cpu().data[i]

        b, k, u, v = 1, *gradients.shape

        
        alpha_num = gradients.pow(2)

        # 先乘后加（原实现应该有误。根据issue:https://github.com/1Konny/gradcam_plus_plus-pytorch/issues/2）
        # alpha_denom = gradients.pow(2).mul(2) + \
        #         activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        
        # 先加后乘（这个实现应该对）
        alpha_denom = gradients.pow(2).mul(2) + \
                    (conv_output.view(b, k, u*v).sum(-1, keepdim=True).view(b,k,1,1)).mul(gradients.pow(3))

        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(grad_cam_pp_pred.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*conv_output).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        # size自行修改。
        saliency_map = F.upsample(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        saliency_maps.append(saliency_map)

        import cv2
        import numpy as np
        saliency_map = np.uint8(255 * saliency_map)
        heatmap = cv2.applyColorMap(cv2.resize(saliency_map[0,0],(224, 224)), cv2.COLORMAP_JET)
        cv2.imwrite("hey.jpg",heatmap)

    saliency_maps = torch.stack(saliency_maps).cuda()

    return saliency_maps




if __name__ == "__main__":
    from models.deep_learning import cnn

    # themodel = nn.Sequential(
    #     cnn.CNNExtractor(3),
    #     cnn.CNNClassifier()
    # )
    # print(themodel[1])

    themodel = cnn.CNN()

    grad_cam(themodel, torch.rand((2,3,64,64)))

    # guided_bp(themodel, torch.rand((2,3,64,64)))

    # grad_cam_pp(themodel, torch.rand((2,3,64,64)))

    print("a")