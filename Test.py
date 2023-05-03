import matplotlib.pyplot as plt
from Config import *
import numpy as np
import torch
from torch import Tensor



def image2tensor(image) -> Tensor:
    tensor = torch.from_numpy(np.array(image, np.float32, copy=False))
    return tensor

def PercentileRescaler(Arr):
    minval=np.percentile(Arr, 0, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
    maxval=np.percentile(Arr, 100, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)

    if minval==maxval:
        print("min=max")
    Arr=(Arr-minval)/(maxval-minval)
    Arr=np.clip(Arr, 0.0, 1.0)
    return Arr, minval, maxval

def RestoreRescaler(Arr,minval,maxval):
    arr=Arr*(maxval - minval)+ minval
    arr = np.clip(arr, 0.0, maxval)
    return arr


def main() -> None:
    generator.eval()


    filenames = os.listdir(lr_dir)
    total_files = len(filenames)


    for index in range(total_files):
        print (filenames[index])

        lr_path = os.path.join(lr_dir, filenames[index])
        lr_image = np.load(lr_path)
        lr_image, minVal, maxVal= PercentileRescaler(lr_image)

        hr_tensor = image2tensor(lr_image).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():


            sr_tensor = generator(hr_tensor)
            sr_tensor = sr_tensor.squeeze()
            img = torch.from_numpy(np.array(sr_tensor.to('cpu'), np.float32, copy=False))
            sr_image = np.array(img)
            sr_image= np.clip (sr_image,0.0,1.0)
            sr_image = RestoreRescaler(np.array(sr_image), minVal, maxVal)

        plt.imsave(sr_dir + filenames[index] + ".tiff", sr_image, cmap='gray')

if __name__ == "__main__":
    main()
