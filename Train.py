
import time

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.fft

import numpy as np

from Config import *
from Dataset import CustomDataset
from torch.autograd import Variable

def main ():
    print('Job Start')
    print (torch.__version__)
    
    discriminator.train()
    generator.train()

    dataset = CustomDataset(dataroot)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, True, pin_memory=True)
    num_batches = len(dataloader)

    for epoch in range(start_g_epoch, g_epochs):
        for index, data in enumerate(dataloader):

            # Configure model input
            imgs_lr = Variable(data[0].type(Tensor))
            imgs_hr = Variable(data[1].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)


            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)
            loss_FFT=criterion_FFT(torch.fft.rfft2(gen_hr),torch.fft.rfft2(imgs_hr))

            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr).detach()
            loss_content = criterion_content(gen_features, real_features)


            # Total generator loss
            loss_G = loss_content + lambda_adv * loss_GAN + lambda_pixel * loss_pixel + lambda_fft*loss_FFT

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f, FFT: %f]"
                % (
                    epoch,
                    g_epochs,
                    index,
                    num_batches,
                    loss_D.item(),
                    loss_G.item(),
                    loss_content.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
					loss_FFT.item(),
                )
            )



        torch.save(discriminator.state_dict(), os.path.join("samples", f"D_epoch{epoch}.pth"))
        torch.save(generator.state_dict(), os.path.join("samples", f"G_epoch{epoch}.pth"))

    # Save final model.
    torch.save(discriminator.state_dict(), os.path.join("results", "D-last.pth"))
    torch.save(generator.state_dict(), os.path.join("results", "G-last.pth"))

    print('Job Done')


if __name__ == "__main__":
    main()