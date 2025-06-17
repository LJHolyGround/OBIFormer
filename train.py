##### External Interface #####
import argparse
import datetime
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from datasets import *
from models.OBIFormer import CharFormer
##### Internal Interface #####
from util.LossFunctions import PSNRLoss, VGGPerceptualLoss
from util.TestMetrics import get_PSNR, get_SSIM

##### Optional Tools #####


def TrainModel(opt):
    ##### Determine CUDA #####
    cuda = True if torch.cuda.is_available() else False
    # device = torch.device("cuda" if cuda else "cpu")

    ##### Initialize models: generator and discriminator #####
    model = CharFormer(dim=16, stages=opt.stages, depth_RSAB=opt.depth_RSAB,
                       depth_GSNB=opt.depth_GSNB, dim_head=64, heads=8)
    if cuda:
        model = model.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        model.load_state_dict(torch.load(opt.checkpoint, map_location='cuda'))

    ##### Print Model Size #####
    # DNN_printer(model, (opt.channels, opt.img_height, opt.img_width), opt.batch_size)

    ##### Define Optimizers #####
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    ##### Configure Dataloaders #####
    # data transformations
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])]

    # training data
    dataloader = DataLoader(
        ImageDataset(opt.train_input, transforms_=transforms_, file_set="separate",
                     root2=opt.train_target),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # validating data
    val_dataloader = DataLoader(
        ImageDataset(opt.val_input, transforms_=transforms_, mode="val", file_set="separate",
                     root2=opt.val_target),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    ########################################
    #####      Parameter Setting       #####
    ########################################

    ##### Loss functions #####
    criterion_pixelwise = PSNRLoss()
    criterion_VGG = VGGPerceptualLoss()
    if cuda:
        criterion_pixelwise.cuda()
        criterion_VGG.cuda()

    ##### Loss weight of L1 pixel-wise loss between translated image and real image #####
    lambda_pixel = 100
    lambda_vgg = 0.05
    lamdba_sk = 100

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    ########################################
    #####      Tools Definitions       #####
    ########################################

    ##### calculate the evaluation metrics on validation set #####
    def output_metrics():
        PSNR = SSIM = 0
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                input = Variable(batch["B"].type(Tensor))
                target = Variable(batch["A"].type(Tensor))
                output, features_ = model(input)

                ndarr_target = make_grid(target.data).mul(255).add_(0.5).clamp_(
                    0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                ndarr_output = make_grid(output.data).mul(255).add_(0.5).clamp_(
                    0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

                PSNR += get_PSNR(ndarr_target, ndarr_output)
                SSIM += get_SSIM(ndarr_target, ndarr_output)

        return PSNR / len(val_dataloader), SSIM / len(val_dataloader)

    ########################################
    #####           Training           #####
    ########################################

    prev_time = time.time()
    best_psnr = best_ssim = 0
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            ##### Model Inputs: input pics and target pics #####
            input = Variable(batch["B"].type(Tensor))
            target = Variable(batch["A"].type(Tensor))
            target_sk = Variable(batch["F"].type(Tensor))

            # ------------------
            #  Train Generators
            # ------------------

            ##### Generator Losses #####
            output, feature_ = model(input)
            loss_pixel = criterion_pixelwise(output, target)
            loss_VGG = criterion_VGG(output, target)

            loss_skeleton = criterion_pixelwise(feature_, target_sk)
            loss_VGG_sk = criterion_VGG(feature_, target_sk)

            ##### Total Generator Loss #####
            loss_total = lambda_pixel * loss_pixel + lambda_vgg * \
                loss_VGG + lamdba_sk * loss_skeleton + loss_VGG_sk

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # --------------
            #  Log Progress
            # --------------

            ##### Approximate finishing time #####
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            ##### Print log #####
            if i % 400 == 0:
                print("\r[EPOCH %d/%d] [BATCH %d/%d] [G LOSS: %f, PIXEL: %f,  VGG: %f] EstimateTime: %s"
                      % (
                          epoch,
                          opt.n_epochs,
                          i,
                          len(dataloader),
                          loss_total.item(),
                          loss_pixel.item(),
                          loss_VGG.item(),
                          time_left,
                      )
                      )

        ##### Optional logs in each epoch (by wandb) #####
        PSNR, SSIM = output_metrics()

        ##### Save model checkpoints #####
        if not os.path.exists(dir):
            os.makedirs(dir)
        if PSNR > best_psnr:
            print('Saving Best PSNR Model!')
            best_psnr = PSNR
            torch.save(model.state_dict(
            ), "{}/best_psnr.pth".format(opt.store_path))

        if SSIM > best_ssim:
            print('Saving Best SSIM Model!')
            best_ssim = SSIM
            torch.save(model.state_dict(
            ), "{}/best_ssim.pth".format(opt.store_path))
        print("LOSS: %s, PSNR: %s, SSIM: %s, BEST_PSNR: %s, BEST_SSIM: %s" %
              (loss_total.item(), PSNR, SSIM, best_psnr, best_ssim))


if __name__ == '__main__':
    ##### arguments settings #####
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", type=float, default=100, help="a1")
    parser.add_argument("--a2", type=float, default=0.05, help="a2")
    parser.add_argument("--a3", type=float, default=100, help="a3")
    parser.add_argument("--a4", type=float, default=1, help="a4")
    parser.add_argument("--epoch", type=int, default=0,
                        help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="number of epochs of training")
    parser.add_argument("--train_input", type=str,
                        default="/path/to/input/of/train", help="path to input of train")
    parser.add_argument("--train_target", type=str,
                        default="/path/to/target/of/train", help="path to target of train")
    parser.add_argument("--val_input", type=str,
                        default="/path/to/input/of/val", help="path to input of val")
    parser.add_argument("--val_target", type=str,
                        default="/path/to/target/of/val", help="path to target of val")
    parser.add_argument("--checkpoint", type=str,
                        default="/path/to/checkpoint", help="path to checkpoint")
    parser.add_argument("--store_path", type=str,
                        default="/path/to/results", help="path to results")
    parser.add_argument("--batch_size", type=int,
                        default=10, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--stages", type=int, default=4,
                        help="stages of U net")
    parser.add_argument("--depth_RSAB", type=int, default=2,
                        help="number of transformer per RSAB")
    parser.add_argument("--depth_GSNB", type=int, default=2,
                        help="number of Conv2d per GSNB")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int,
                        default=256, help="size of image height")
    parser.add_argument("--img_width", type=int,
                        default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels")
    opt = parser.parse_args()

    ##### Run training process #####
    TrainModel(opt)
