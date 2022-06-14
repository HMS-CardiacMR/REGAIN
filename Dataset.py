from typing import Tuple
import torch.utils.data as data
import torchvision.transforms
from Config import *
import numpy as np


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, dataroot: str) -> None:
        super(CustomDataset, self).__init__()
        # Get the index of all images in the high-resolution folder and
        # low-resolution folder under the data set address.
        # Note: The high and low resolution file index should be corresponding.
        lr_dir_path = os.path.join(dataroot, "LR_Mag")
        hr_dir_path = os.path.join(dataroot, "HR_Mag")

        lr_filenames = os.listdir(lr_dir_path)
        hr_filenames = os.listdir(hr_dir_path)

        self.lr_filenames = [os.path.join(lr_dir_path, x) for x in lr_filenames]
        self.hr_filenames = [os.path.join(hr_dir_path, x) for x in hr_filenames]

    def transform(self, lr_image, hr_image, resample):
        lr_tensor = torchvision.transforms.functional.to_pil_image(lr_image)
        hr_tensor = torchvision.transforms.functional.to_pil_image(hr_image)
        # Random crop

        self.crop_indices = torchvision.transforms.RandomCrop.get_params(lr_tensor, output_size=(image_size, image_size))

        i, j, h, w =  self.crop_indices



        lr_tensor=torchvision.transforms.functional.crop(lr_tensor,i, j, h, w)
        hr_tensor = torchvision.transforms.functional.crop(hr_tensor, i, j, h, w)

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            lr_tensor = torchvision.transforms.functional.hflip(lr_tensor)
            hr_tensor = torchvision.transforms.functional.hflip(hr_tensor)

        # Random vertical flipping
        if torch.rand(1) > 0.5:
            lr_tensor = torchvision.transforms.functional.vflip(lr_tensor)
            hr_tensor = torchvision.transforms.functional.vflip(hr_tensor)


        makeTensor= torchvision.transforms.ToTensor()
        lr_tensor = makeTensor(lr_tensor)
        hr_tensor = makeTensor(hr_tensor)
        return lr_tensor, hr_tensor

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        lr = np.load(self.lr_filenames[index])
        hr = np.load(self.hr_filenames[index])

        lr = torch.from_numpy(np.array(lr, np.float32, copy=False))
        hr = torch.from_numpy(np.array(hr, np.float32, copy=False))

        lrts, hrts = self.transform(lr, hr, True)


        return lrts, hrts

    def __len__(self) -> int:
        return len(self.lr_filenames)