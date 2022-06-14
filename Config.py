import os
import datetime
import torch.backends.cudnn as cudnn
from Models import *
# ==============================================================================
#                              Common configure
# ==============================================================================
torch.manual_seed(123)
cudnn.benchmark = True
cudnn.deterministic = False

use_gpu=False

if use_gpu:
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
else :
    device = torch.device("cpu")

# Runing mode.
mode = "validate"

# ==============================================================================
#                              Train configure
# ==============================================================================
if mode == "train":

    # 1. Dataset path.
    dataroot = "/dataroot"

    image_size=image_In
    batch_size = 64

    # 2. Define model.
    generator = GeneratorRRDB(1,filters=64, num_res_blocks=23).to(device)
    discriminator = Discriminator(input_shape=(1, image_size, image_size)).to(device)
    feature_extractor = FeatureExtractor().to(device)

    # 3. Resume training.
    seed=123
    start_p_epoch = 0
    start_g_epoch = 0
    resume = False
    resume_p_weight = ""
    resume_d_weight = ""
    resume_g_weight = ""
    
    if resume:
        discriminator.load_state_dict(torch.load("./samples/"+resume_d_weight, map_location=device))
        generator.load_state_dict(torch.load("./samples/"+resume_g_weight, map_location=device))


    # 4. Number of epochs.
    g_epochs = 750

    # 5. Loss function.
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.MSELoss().to(device)
    criterion_FFT = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.MSELoss().to(device)


    # Loss function weight.
    lambda_adv = 5e-03
    lambda_pixel = 1e-02
    lambda_fft = 1e-02

    # 6. Optimizer.
    p_lr = 1e-4
    d_lr = 1e-4
    g_lr = 1e-4

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor



# ==============================================================================
#                              Validate configure
# ==============================================================================
if mode == "validate":
    dataroot = "/dataroot"

    image_size = image_In

    generator = GeneratorRRDB(1, filters=64, num_res_blocks=23).to(device)
    discriminator = Discriminator(input_shape=(1, image_size, image_size)).to(device)

    generator.load_state_dict(torch.load("./PretrainedModel/Pretrained_Model_REGAIN_epoch500.pth", map_location=device))

    lr_dir = dataroot+"/Test_LR_mag/"
    hr_dir = dataroot+"/Test_HR_mag/"
    sr_dir = dataroot+"/Out_SR_mag/"