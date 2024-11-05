from PIL import Image
import torch.nn.functional
import torchvision.transforms.functional
import numpy as np
import cv2

impath = './Pose27_64x64_files/'
mode_chosen ="area"
idx = 0
for p in range(3329):
    img = Image.open(impath+str(p+1)+'.jpg')
    img = torchvision.transforms.functional.to_tensor(img) # 'PIL.JpegImagePlugin.JpegImageFile' -> 'torch.Tensor' gray: [1,H,W]
    img = img.unsqueeze(0) # [c,H,W] -> [1,c,H,W]

    img12 = torch.nn.functional.interpolate(img, scale_factor=0.5, mode=mode_chosen) # [B,C,H,W]: batch_size * channel * H*W
    img12 = img12.squeeze(0) # [1,1,H,W] -> [1,H,W]
    img12 = img12.squeeze(0) # [1,H,W] -> [H,W]
    img12 = img12 * 255
    idx += 1
    save_dir = "./PIE_C27_32_"+mode_chosen+"/"+str(idx)+".jpg"
    cv2.imwrite(save_dir, np.array(img12))
