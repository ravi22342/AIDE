
import sys
import os
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
# sys.path.extend(['../'])
import os, time, argparse, random
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import pandas as pd
# import pydicom
# from skimage import measure
import torch.nn.functional as F
# import shutil
# import SimpleITK as sitk
from datasetprostate_proposed import prostate_seg, Compose, Resize, RandomRotate, RandomHorizontallyFlip, ToTensor, \
    Normalize
from models_singlemodalinput import UNet, UNetsa
from utils import PolyLR
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description='Segmeantation for Prostate',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', default='UNet', type=str, help='UNet, ...')
    parser.add_argument('--data_mean', default=None, nargs='+', type=float,
                        help='Normalize mean')
    parser.add_argument('--data_std', default=None, nargs='+', type=float,
                        help='Normalize std')
    parser.add_argument('--rotation', default=60, type=float, help='rotation angle')
    parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
    parser.add_argument('--gpu_order', default='0', type=str, help='gpu order')
    parser.add_argument('--torch_seed', default=2, type=int, help='torch_seed')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--warmup_epoch', default=1, type=int, help='pretrain num epoch')
    parser.add_argument('--num_epoch', default=50, type=int, help='num epoch')
    parser.add_argument('--loss', default='cedice', type=str, help='ce, dice')
    parser.add_argument('--img_size', default=256, type=int, help='512')
    parser.add_argument('--temperature', default=1.0, type=float, help='0.5')
    parser.add_argument('--lr_policy', default='StepLR', type=str, help='StepLR')
    parser.add_argument('--cedice_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='weight for ce and dice loss')
    parser.add_argument('--segcor_weight', default=[1.0, 10.0], nargs='+', type=float,
                        help='weight for seg and pseudolabel seg')
    parser.add_argument('--ceclass_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='categorical weight for ce loss')
    parser.add_argument('--diceclass_weight', default=[1.0, 1.0], nargs='+', type=float,
                        help='categorical weight for dice loss')
    parser.add_argument('--checkpoint', default='checkpoint_traindxgenerate3t_comparisoncrossdomain/')
    parser.add_argument('--history', default='history_traindxgenerate3t_comparisoncrossdomain')
    parser.add_argument('--cudnn', default=0, type=int, help='cudnn')
    parser.add_argument('--repetition', default=100, type=int, help='...')
    parser.add_argument('--tensorboard_path', default="/nfs1/sutrave/AIDE/data/model/tensorbaord")

    args = parser.parse_args()
    return args


def makefolder(folderpath):
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)


def record_params(args):
    localtime = time.asctime(time.localtime(time.time()))
    logging.info('Segmeantation for Prostate MR(Data: {}) \n'.format(localtime))
    logging.info('**************Parameters***************')

    args_dict = args.__dict__
    for key, value in args_dict.items():
        logging.info('{}: {}'.format(key, value))
    logging.info('**************Parameters***************\n')


def build_model(model_name, num_classes):
    if model_name.lower() == 'unet':
        net = UNet(num_classes=num_classes)
    elif model_name.lower() == 'unetsa':
        net = UNetsa(num_classes=num_classes)
    else:
        raise ValueError('Model not implemented')
    return net


def reverseaug(augset, augoutput, classno):
    for batch_idx in range(len(augset['augno'])):
        for aug_idx in range(augset['augno'][batch_idx]):
            imgflip = augset['hflip{}'.format(aug_idx + 1)][batch_idx]
            rotation = 0 - augset['degree{}'.format(aug_idx + 1)][batch_idx]
            for classidx in range(classno):
                mask = augoutput[aug_idx][batch_idx, classidx, :, :]
                mask = mask.cpu().numpy()
                mask = Image.fromarray(mask, mode='F')
                if imgflip:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.rotate(rotation, Image.BILINEAR)
                mask = torch.from_numpy(np.array(mask))
                augoutput[aug_idx][batch_idx, classidx, :, :] = mask
    return augoutput


def sharpen(mask, temperature):
    masktemp = torch.pow(mask, temperature)
    masktempsum = masktemp.sum(dim=1).unsqueeze(dim=1)
    sharpenmask = masktemp / masktempsum
    return sharpenmask


def get_dice_score(inputs, targets, smooth=1):
    N = targets.size(0)
    targets = targets.view(N, 1, 256, 256)
    inputs = torch.sigmoid(inputs)
    threshold = torch.tensor([0.5])
    inputs = (inputs > threshold) * 1
    iflat = inputs.reshape(-1)
    tflat = targets.reshape(-1)
    intersection = (iflat * tflat).sum()
    dice = (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    return dice


def Dice_fn(inputs, targets, threshold=0.5):
    inputs = F.sigmoid(inputs)
    dice = 0.
    img_count = 0
    for input_, target_ in zip(inputs, targets):
        iflat = input_.view(-1).float()
        tflat = target_.view(-1).float()
        intersection = (iflat * tflat).sum()
        dice_single = ((2. * intersection) / (iflat.sum() + tflat.sum()))
        img_count += 1
        dice += dice_single
    return dice


class consistencyLossImage(nn.Module):

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        mseloss = torch.nn.MSELoss(reduction="mean")
        return mseloss(inputs, targets)


class segmentationLossImage(nn.Module):

    def forward(self, inputs, targets):
        torch.save(inputs, 'inputs.pt')
        torch.save(targets, 'targets.pt')

        N = targets.size(0)
        inputs = inputs.view(N, 256, 256)
        # dice loss
        dice_inputs = inputs[:]
        dice_inputs = torch.sigmoid(dice_inputs)
        # flatten
        iflat = dice_inputs.view(N, -1)
        tflat = targets.view(N, -1)
        intersection = (iflat * tflat).sum(1)
        dice = (2. * intersection + 1) / (iflat.sum(1) + tflat.sum(1) + 1)
        dice_loss = 1 - dice
        # Binary cross entropy
        BCE_logits_loss = torch.nn.BCEWithLogitsLoss()
        BCE_loss = BCE_logits_loss(inputs, targets.float())
        return dice_loss + (1 * BCE_loss)


def Train(train_root, train_csv, test_csv, traincase_csv, testcase_csv, labelcase_csv, tempmaskfolder):
    makefolder(os.path.join(train_root, tempmaskfolder))

    # parameters
    args = parse_args()

    # record
    record_params(args)

    train_cases = pd.read_csv(traincase_csv)['Image'].tolist()
    train_masks = pd.read_csv(traincase_csv)['Mask'].tolist()
    test_cases = pd.read_csv(testcase_csv)['Image'].tolist()
    label_cases = pd.read_csv(labelcase_csv)['Image'].tolist()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_order
    torch.manual_seed(args.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)
    np.random.seed(args.torch_seed)
    random.seed(args.torch_seed)

    if args.cudnn == 0:
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
        cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 1
    net1 = build_model(args.model_name, num_classes)
    net2 = build_model(args.model_name, num_classes)
    params1_name = '{}_temp{}_r{}_net1.pkl'.format(args.model_name, args.temperature, args.repetition)
    params2_name = '{}_temp{}_r{}_net2.pkl'.format(args.model_name, args.temperature, args.repetition)

    start_epoch = 0
    end_epoch = args.num_epoch

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net1 = nn.DataParallel(net1)
        net2 = nn.DataParallel(net2)
    net1.to(device)
    net2.to(device)

    # data
    train_aug = Compose([
        Resize(size=(args.img_size, args.img_size)),
        RandomRotate(args.rotation),
        RandomHorizontallyFlip(),
        ToTensor(),
        Normalize(mean=args.data_mean,
                  std=args.data_std)])
    test_aug = Compose([
        Resize(size=(args.img_size, args.img_size)),
        ToTensor(),
        Normalize(mean=args.data_mean,
                  std=args.data_std)])

    train_dataset = prostate_seg(root=train_root, csv_file=train_csv, tempmaskfolder=tempmaskfolder,
                                 transform=train_aug)
    test_dataset = prostate_seg(root=train_root, csv_file=test_csv, tempmaskfolder=tempmaskfolder, transform=test_aug)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=4, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=4, shuffle=False)

    rate_schedule = np.ones(args.num_epoch)

    optimizer1 = Adam(net1.parameters(), lr=args.lr, amsgrad=True)
    optimizer2 = Adam(net2.parameters(), lr=args.lr, amsgrad=True)

    ## scheduler
    if args.lr_policy == 'StepLR':
        scheduler1 = StepLR(optimizer1, step_size=30, gamma=0.5)
        scheduler2 = StepLR(optimizer2, step_size=30, gamma=0.5)
    if args.lr_policy == 'PolyLR':
        scheduler1 = PolyLR(optimizer1, max_epoch=end_epoch, power=0.9)
        scheduler2 = PolyLR(optimizer2, max_epoch=end_epoch, power=0.9)

    # training process
    logging.info('Start Training For Prostate Seg')
    seg_loss = segmentationLossImage()
    consistency_loss = consistencyLossImage()
    besttraincasedice = 0.0

    for epoch in range(start_epoch, end_epoch):
        ts = time.time()
        rate_schedule[epoch] = min((float(epoch) / float(args.warmup_epoch)) ** 2, 1.0)

        # train
        net1.train()
        net2.train()

        train_loss1 = 0.
        train_dice1 = 0.
        train_count = 0
        train_loss2 = 0.
        train_dice2 = 0.

        for batch_idx, (inputs, augset, targets, targets1, targets2) in \
                tqdm(enumerate(train_loader), total=int(len(train_loader.dataset) / args.batch_size)):

            augoutput1 = []
            augoutput2 = []

            for aug_idx in range(augset['augno'][0]):
                augimg = augset['img{}'.format(aug_idx + 1)].to(device)
                augimg = augimg.unsqueeze(dim=1)
                augoutput1.append(net1(augimg).detach())
                augoutput2.append(net2(augimg).detach())

            augoutput1 = reverseaug(augset, augoutput1, classno=num_classes)
            augoutput2 = reverseaug(augset, augoutput2, classno=num_classes)

            for aug_idx in range(len(augoutput1)):
                augoutput1[aug_idx] = augoutput1[aug_idx].view(targets1.size(0), 256, 256)

            for aug_idx in range(len(augoutput2)):
                augoutput2[aug_idx] = augoutput2[aug_idx].view(targets2.size(0), 256, 256)

            for aug_idx in range(augset['augno'][0]):
                augmask1 = torch.nn.functional.sigmoid(augoutput1[aug_idx])
                augmask2 = torch.nn.functional.sigmoid(augoutput2[aug_idx])

                if aug_idx == 0:
                    pseudo_label1 = augmask1
                    pseudo_label2 = augmask2
                else:
                    pseudo_label1 += augmask1
                    pseudo_label2 += augmask2

            pseudo_label1 = pseudo_label1 / float(augset['augno'][0])
            pseudo_label2 = pseudo_label2 / float(augset['augno'][0])
            pseudo_label1 = sharpen(pseudo_label1, args.temperature)
            pseudo_label2 = sharpen(pseudo_label2, args.temperature)

            weightmap1 = 1.0 - 4.0 * pseudo_label1
            weightmap2 = 1.0 - 4.0 * pseudo_label2

            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            inputs = inputs.unsqueeze(dim=1)
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)

            outputs1 = outputs1.view(targets1.size(0), 256, 256)
            outputs2 = outputs2.view(targets2.size(0), 256, 256)

            loss1_segpre = seg_loss(outputs1, targets2)
            loss2_segpre = seg_loss(outputs2, targets1)

            _, indx1 = loss1_segpre.sort()
            _, indx2 = loss2_segpre.sort()

            loss1_seg1 = seg_loss(outputs1[indx2[0:2], :, :], targets2[indx2[0:2], :, :]).mean()
            loss2_seg1 = seg_loss(outputs2[indx1[0:2], :, :], targets1[indx1[0:2], :, :]).mean()
            loss1_seg2 = seg_loss(outputs1[indx2[2:], :, :], targets2[indx2[2:], :, :]).mean()
            loss2_seg2 = seg_loss(outputs2[indx1[2:], :, :], targets1[indx1[2:], :, :]).mean()

            loss1_cor = weightmap2[indx2[2:], :, :] * consistency_loss(outputs1[indx2[2:], :, :],
                                                                       pseudo_label2[indx2[2:], :, :])
            loss1_cor = loss1_cor.mean()
            loss1 = args.segcor_weight[0] * (loss1_seg1 +  loss1_seg2) + \
                    args.segcor_weight[1] * loss1_cor

            loss2_cor = weightmap1[indx1[2:], :, :] * consistency_loss(outputs2[indx1[2:], :, :],
                                                                       pseudo_label1[indx1[2:], :, :])
            loss2_cor = loss2_cor.mean()
            loss2 = args.segcor_weight[0] * (loss2_seg1 + loss2_seg2) + \
                    args.segcor_weight[1] * loss2_cor

            loss1.backward(retain_graph=True)
            optimizer1.step()
            loss2.backward()
            optimizer2.step()

            train_count += inputs.shape[0]
            train_loss1 += loss1.item() * inputs.shape[0]
            train_dice1 += Dice_fn(outputs1, targets2).item()
            train_loss2 += loss2.item() * inputs.shape[0]
            train_dice2 += Dice_fn(outputs2, targets1).item()

        train_loss1_epoch = train_loss1 / float(train_count)
        train_dice1_epoch = train_dice1 / float(train_count)
        train_loss2_epoch = train_loss2 / float(train_count)
        train_dice2_epoch = train_dice2 / float(train_count)

        print(rate_schedule[epoch])
        print(args.segcor_weight[0] * (loss1_seg1 + loss1_seg2))
        print(args.segcor_weight[1] * loss1_cor)

        print(args.segcor_weight[0] * (loss2_seg1 + loss2_seg2))
        print(args.segcor_weight[1] * loss2_cor)
        print("epoch",epoch)
        print(train_loss1_epoch)
        print(train_dice1_epoch)
        print(train_loss2_epoch)
        print(train_dice2_epoch)

        writer_training.add_scalar("train_loss1",train_loss1_epoch,epoch + 1)
        writer_training.add_scalar("train_dice1",train_dice1_epoch,epoch + 1)
        writer_training.add_scalar("train_loss2",train_loss2_epoch,epoch + 1)
        writer_training.add_scalar("train_dice2",train_dice2_epoch,epoch + 1)
        writer_training.flush()


if __name__ == "__main__":
    args = parse_args()
    train_root = 'D:\Deep_Learning\AIDE\data_files'
    train_csv = 'D:\\Deep_Learning\\AIDE\\data_files\\csvFiles\\train_data2.csv'
    test_csv = 'D:\\Deep_Learning\\AIDE\\data_files\\csvFiles\\test_data1.csv'
    traincase_csv = 'D:\\Deep_Learning\\AIDE\\data_files\\csvFiles\\train_cases2.csv'
    testcase_csv = 'D:\\Deep_Learning\\AIDE\\data_files\\csvFiles\\test_case1.csv'
    labeledcase_csv = 'D:\\Deep_Learning\\AIDE\\data_files\\csvFiles\\labelled_case1.csv'
    tempmaskfolder = 'train_aug'
    makefolder(os.path.join(train_root, tempmaskfolder))
    tempmaskfolder = 'train_aug\{}_{}'.format(args.model_name, args.repetition)
    if not os.path.exists(args.tensorboard_path):
        os.mkdir(args.tensorboard_path)
    writer_training = SummaryWriter(args.tensorboard_path)
    Train(train_root, train_csv, test_csv, traincase_csv, testcase_csv, labeledcase_csv, tempmaskfolder)



if __name__ == "__main__":
    args = parse_args()
    train_root = 'D:\Deep_Learning\AIDE\data_files'
    train_csv = 'D:\\Deep_Learning\\AIDE\\data_files\\csvFiles\\train_data2.csv'
    test_csv = 'D:\\Deep_Learning\\AIDE\\data_files\\csvFiles\\test_data1.csv'
    traincase_csv = 'D:\\Deep_Learning\\AIDE\\data_files\\csvFiles\\train_cases2.csv'
    testcase_csv = 'D:\\Deep_Learning\\AIDE\\data_files\\csvFiles\\test_case1.csv'
    labeledcase_csv = 'D:\\Deep_Learning\\AIDE\\data_files\\csvFiles\\labelled_case1.csv'
    tempmaskfolder = 'train_aug'
    makefolder(os.path.join(train_root, tempmaskfolder))
    tempmaskfolder = 'train_aug\{}_{}'.format(args.model_name, args.repetition)
    # if not os.path.exists(args.tensorboard_path):
    #   os.mkdir(args.tensorboard_path)
    # writer_training = SummaryWriter(args.tensorboard_path)
    Train(train_root, train_csv, test_csv, traincase_csv, testcase_csv, labeledcase_csv, tempmaskfolder)
