import argparse

import torch
from skimage import img_as_ubyte
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from datasets import *
from models.OBIFormer import CharFormer
from util.TestMetrics import get_PSNR, get_SSIM

# from datasets import skeletonPrepare

########################################
#####      Tools Definitions       #####
########################################

##### Load weights of the model #####


def loadModel(model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(opt.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


##### Concat input and target pics for visual comparison #####
def concat(imgA, imgB):
    size1, size2 = imgA.size, imgB.size

    joint = Image.new(opt.color_set, (size1[0] + size2[0], size1[1]))
    loc1, loc2 = (0, 0), (size1[0], 0)
    joint.paste(imgA, loc1)
    joint.paste(imgB, loc2)
    # joint.show()
    return joint


##### Image pre-processing #####
def AdditionalProcess(pic):
    # If input pic is the combination of input and target pics
    if opt.IsPre == True:
        width, height = pic.size
        pic = pic.crop((width / 2, 0, width, height))
    Ori_width, Ori_height = pic.size

    # If there is additional processing on input pics
    # pic = skeletonPrepare(pic)
    return pic, Ori_width, Ori_height

##### Save a given Tensor into an image file #####


def Get_tensor(tensor, nrow=8, padding=2,
               normalize=False, irange=None, scale_each=False, pad_value=0):

    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, scale_each=scale_each)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


if __name__ == '__main__':

    ##### arguments settings #####
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="path/to/input", help="path to input")
    parser.add_argument("--store_path", type=str, default="path/to/results",
                        help="path to results")
    parser.add_argument("--checkpoint", type=str, default="path/to/checkpoint",
                        help="path to checkpoint")
    parser.add_argument("--depth_RSAB", type=int, default=2,
                        help="number of transformer per RSAB")
    parser.add_argument("--depth_GSNB", type=int, default=2,
                        help="number of Conv2d per GSNB")
    parser.add_argument("--img_height", type=int,
                        default=256, help="size of image height")
    parser.add_argument("--img_width", type=int,
                        default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels")
    parser.add_argument("--color_set", type=str, default="RGB",
                        help="number of image color set, RGB or L ")
    parser.add_argument("--IsPre", type=bool, default=True,
                        help="If need to make pre-processing on input pics")
    opt = parser.parse_args()

    ##### Create store_path #####
    os.makedirs(opt.store_path, exist_ok=True)

    ##### Initialize CUDA and Tensor #####
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    ##### data transformations #####
    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])]
    transform = transforms.Compose(transforms_)

    model = loadModel(CharFormer(dim=16, stages=4, depth_RSAB=opt.depth_RSAB,
                      depth_GSNB=opt.depth_GSNB, dim_head=64, heads=8))

    ##### Process input pics #####
    test_dataloader = DataLoader(
        ImageDataset(opt.input_path, transforms_=transforms_, mode="test", file_set="separate",
                     root2=opt.input_path),
        batch_size=10,
        shuffle=True,
        num_workers=8,
    )

    PSNR = SSIM = 0
    for i in range(1):
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                input = Variable(batch["B"].type(Tensor))
                target = Variable(batch["A"].type(Tensor))
                output, feature = model(input)

                ndarr_target = make_grid(target).mul(255).add_(0.5).clamp_(
                    0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                ndarr_output = make_grid(output).mul(255).add_(0.5).clamp_(
                    0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

                dst = opt.store_path
                for j in range(input.shape[0]):
                    img_name = os.path.basename(batch["target_path"][j])[:-4]
                    save_file_1 = os.path.join(dst, f'{img_name}_lq.jpg')
                    save_file_2 = os.path.join(dst, f'{img_name}_gt.jpg')
                    save_file_3 = os.path.join(dst, f'{img_name}_sk.jpg')
                    save_file_4 = os.path.join(
                        opt.store_path, f'{img_name}.jpg')
                    cv2.imwrite(save_file_1, img_as_ubyte(
                        input[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy()))
                    cv2.imwrite(save_file_2, img_as_ubyte(
                        target[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy()))
                    cv2.imwrite(save_file_3, img_as_ubyte(
                        feature[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy()))
                    cv2.imwrite(save_file_4, img_as_ubyte(
                        output[j].clamp(0, 1).permute(1, 2, 0).cpu().numpy()))

                PSNR += get_PSNR(ndarr_target, ndarr_output)
                SSIM += get_SSIM(ndarr_target, ndarr_output)
    print(PSNR / len(test_dataloader) / 1, SSIM / len(test_dataloader) / 1)
