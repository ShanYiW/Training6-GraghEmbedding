import numpy as np
from PIL import Image
import torch.nn.functional
import torchvision.transforms.functional
import cv2

impath = "./att_faces/"
ratio = 0.5
orl_mode ="area"
idx = 0
for person in range(40):
    for pose in range(10):
        img = Image.open(impath+'s'+str(person+1)+'\\'+str(pose+1)+'.pgm')
        img = torchvision.transforms.functional.to_tensor(img) # 'PIL.JpegImagePlugin.JpegImageFile' -> 'torch.Tensor' 灰度: [1,H,W]
        img = img.unsqueeze(0) # [c,H,W] -> [1,c,H,W]

        img_down = torch.nn.functional.interpolate(img, scale_factor=ratio, mode=orl_mode) # [B,C,H,W]: batch_size * channel * H*W
        img_down = img_down.squeeze(0) # [1,1,H,W] -> [1,H,W]
        img_down = img_down.squeeze(0) # [1,H,W] -> [H,W]
        img_down = img_down * 255
        idx += 1
        save_dir = "./ORL_12_"+orl_mode+"/orl"+str(idx)+".pgm"
        cv2.imwrite(save_dir, np.array(img_down))