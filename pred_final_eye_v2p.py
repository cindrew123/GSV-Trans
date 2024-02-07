import argparse
import os
import itertools

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable, grad
import torch.optim as optim
# import scipy.misc
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor
from sklearn.metrics import f1_score

from tensorboardX import SummaryWriter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


import pickle
import cv2
import sys
from tqdm import tqdm
from PIL import Image


from deeplab.final_eye_model import VisionTransformer as ViT_seg
from deeplab.final_eye_model import CONFIGS as CONFIGS_ViT_seg
from model.discriminator import Discriminator_NumClass_2
from dataset.isprs_vai import ISPRS_VAI
from dataset.isprs_vai_TwoImage import ISPRS_VAI_TwoImage
from dataset.isprs_pot_irrg import ISPRS_POT_IRRG
from dataset.isprs_pot_rgb import ISPRS_POT_RGB
from dataset.isprs_pot_rgb_TwoImage import ISPRS_POT_RGB_TwoImage
from dataset.isprs_pot_irrg_TwoImage import ISPRS_POT_IRRG_TwoImage
import random
import timeit

start = timeit.default_timer()

# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IMG_MEAN_BEIJING = np.array((82.90118227, 99.76638055, 96.33826206), dtype=np.float32)
IMG_MEAN_ISPRS = np.array((79.78101224, 81.65397782, 120.85911836), dtype=np.float32)

BATCH_SIZE = 2
DATA_DIRECTORY = '../../datasets/remote_beijing'
DATA_LIST_PATH = '../../datasets/remote_beijing/ImageSets/train.txt'
VAL_DATA_LIST_PATH = '../../datasets/remote_beijing/ImageSets/val.txt'

NUM_CLASSES = 6

# DATA_DIRECTORY = '../../datasets/VOCdevkit/VOC2012'
# DATA_LIST_PATH = '../../datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
# NUM_CLASSES = 21
IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4

MOMENTUM = 0.9

NUM_STEPS = 100000
POWER = 0.9
RANDOM_SEED = 1234
# RESTORE_FROM = '../../datasets/MS_DeepLab_resnet_pretrained_COCO_init.pth'
#RESTORE_FROM = './experiments/ckpt/train_potirrg2vai_colorjitter_fixed_lr_withMSELoss_2_ssl_larger_ma_dual_mean_teacher_from_without_g_siamese_change_numclass_2_newUpsample/model_cheakpoint_model_1_for_iter_{}.pth'

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './experiments/restores/move_no_0_v2p'
RESULT_PATH = './experiments/evaluate_result/move_no_0_v2p/'
LOG_PATH = './experiments/TensorboardLogs/move_no_0_v2p'
DIFF_PATH = './experiments/pred_gt_diff/move_no_0_v2p'
PRED_PATH = './experiments/pred_result/move_no_0_v2p'
WEIGHT_DECAY = 0.0005


## gai
RESTORE_FROM = './best_model/vit168000.pth'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--val-data-list", type=str, default=VAL_DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-d", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training epoches.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--result-path", type=str, default=RESULT_PATH,
                        help="Where to save preds result of the model.")
    parser.add_argument("--log-path", type=str, default=LOG_PATH,
                        help="Where to save tensorboard log result of the model.")
    parser.add_argument("--diff-path", type=str, default=DIFF_PATH,
                        help="Where to save pred and gt diff result of the model.")
    parser.add_argument("--pred-path", type=str, default=PRED_PATH,
                        help="Where to save pred result of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
                        
    # add VisionTransformer
    parser.add_argument('--vit_name', type=str,
                        default='ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--img_size', type=int,
                        default=512, help='input patch size of network input')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
                    
    parser.add_argument('--multi_trans_name', type=str,
                        default='basical-transblock', help='select one multi trans model')
    # set the depth of the multitrans
    parser.add_argument('--multi_trans_depth', type=str,
                        default='3', help='select multi trans depth')
    return parser.parse_args()


args = get_arguments()

def colorize_mask(mask, palette):
    # mask: uint8 numpy array of the mask
    new_mask = Image.fromarray(mask).convert('P')
    palette = sum(palette, [])  # a list
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask.putpalette(palette)

    return new_mask

def loss_calc_G(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # label = Variable(label.long()).cuda()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL).cuda()

    return criterion(pred, label)


def loss_calc_D(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # label = Variable(label.long()).cuda()
    criterion = torch.nn.BCELoss().cuda()

    return criterion(pred, label)


def res_loss_calc(real, fake):
    criterion = nn.L1Loss().cuda()

    return criterion(real, fake)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)
    # b.append(model.layer5)


    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            if i.requires_grad:
                yield i


def critic_params(model):
    b = []
    b.append(model.parameters())
    for j in range(len(b)):
        for i in b[j]:
            if i.requires_grad:
                yield i


def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for images, labels, size, name in data_loader:
            yield (images, labels, size, name)


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def calc_gradient_penalty(D, real_data, fake_data):
    """Calculatge gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_data.size())
    # alpha = alpha.expand(args.batch_size, real_data.nelement() / args.batch_size).contiguous().view(real_data.size())
    alpha = alpha.cuda()

    interpolates = make_variable(alpha * real_data + ((1 - alpha) * fake_data))
    interpolates.requires_grad = True

    disc_interpolates, _ = D(interpolates)

    gradients = grad(outputs=disc_interpolates,
                     inputs=interpolates,
                     grad_outputs=make_cuda(
                         torch.ones(disc_interpolates.size())),
                     create_graph=True,
                     retain_graph=True,
                     only_inputs=True)[0]

    gradient_penalty = 10 * \
                       ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def loss_plot(hist, path='Train_hist.png', model_name=''):
    x = hist['steps']

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()


def loss_plot_sigle(hist, path='Train_hist.png', model_name=''):
    x = hist['steps']
    root = path
    for name, value in hist.iteritems():
        if name == 'steps' or name == 'mIOUs':
            continue
        else:
            y = value
            plt.plot(x, y, label=name)
            plt.xlabel('Iter')
            plt.ylabel(name)
            plt.legend(loc=4)
            plt.grid(True)
            plt.tight_layout()

            loss_path = os.path.join(root, model_name + '_' + name + '.png')

            plt.savefig(loss_path)

            plt.close()


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n) & (b < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def per_class_recall(hist):
    return np.diag(hist) / hist.sum(0)

def per_class_accuracy(hist):
    return np.diag(hist) / hist.sum(1)

def compute_mIoU(pred_imgs, gt_imgs, num_classes, save_path):
    """
    Function to compute mean IoU
    Args:
    	pred_imgs: Predictions obtained using our Neural Networks
    	gt_imgs: Ground truth label maps
    	json_path: Path to cityscapes_info.json file
    Returns:
    	Mean IoU score
    """

    hist = np.zeros((num_classes, num_classes))

    for ind in range(len(gt_imgs)):
        pred = pred_imgs[ind]
        label = gt_imgs[ind]
        if len(label.flatten()) != len(pred.flatten()):
            # print(
            # 'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()),
            #                                                                 gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # if ind > 0 and ind % 10 == 0:
        #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.nanmean(per_class_iu(hist))))

    IoUs = per_class_iu(hist)
    accs = per_class_accuracy(hist)
    recalls = per_class_recall(hist)
    mIoUs = np.nanmean(IoUs)*100
    maccs = np.nanmean(accs)*100
    mrecalls = np.nanmean(recalls)*100
    print('===> mIoU: ' + str(mIoUs))
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(mIoUs) + '\n')
            f.write(str(IoUs.tolist()) + '\n')
            f.write('meanacc: ' + str(maccs) + '\n')
            f.write(str(accs.tolist()) + '\n')
            f.write('meanrecall: ' + str(mrecalls) + '\n')
            f.write(str(recalls.tolist()) + '\n')
            f.write(str(hist) + '\n')
    return mIoUs,IoUs
    


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from deeplab.metric2 import ConfusionMatrix
    # print 1
    # print data_list

    ConfM = ConfusionMatrix(class_num)

    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M, class_iou = ConfM.jaccard()
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')

    return aveJ

def mytest(model, step, testloader,device,writer,palette):
    model.eval()
    #interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
    label_trues, label_preds = [], []
    Class_F1 = [0,0,0,0,0]
    count = 0
    Mean_F1 = 0
    np.set_printoptions(threshold=100000000)

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print ('The''iter of ', index, 'processed')
        image, label, size, name = batch
        image = Variable(image, requires_grad=False).cuda(args.gpu)
        size = size[0].numpy()
        output, pseudo_avr, pseudo_vb, pseudo_multi = model(image)
        # output = interp(output).cpu().data[0].numpy()

        upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
        output = upsample(output).cpu().data[0].numpy()

        output = output[:, :size[0], :size[1]]
        gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.uint8)

        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        pred_img = colorize_mask(output, palette)
        name = name[0].strip()

        if index < 300:
            pred_img.save('%s/%s' % (args.pred_path, name + '.tif'))



        gt1 = gt.flatten()
        output1 = output.flatten()
        class_F1 = f1_score(gt1, output1, average=None).tolist()
        mean_F1 = f1_score(gt1, output1, average='weighted')
        # avoid the influence of ignore label and lost label
        if np.sum(gt == 255) != 0:
            class_F1.pop()
            #print ('delete',255)

        if len(class_F1) < 5:
            for i in range(5):
                if np.sum(gt == i) == 0:
                    # print ('lost',i)
                    class_F1.insert(i, 0)
        # if len(class_F1) > 5:
        if len(class_F1) > 5:
            for i in range(len(class_F1) - 5):
                if len(class_F1) > 5:
                    class_F1.pop()
            


        Class_F1 = np.sum([class_F1,Class_F1],axis=0)
        Mean_F1= mean_F1+Mean_F1

        

        if index % 1 == 0:
            pred_col = colorize_mask(output, palette)
            gt_col = colorize_mask(gt, palette)
            # add image to tensorboard
            writer.add_image('pred/' + name[0],
                             ToTensor()(pred_col.convert('RGB')), step)

            writer.add_image('GT/' + name[0],
                         ToTensor()(gt_col.convert('RGB')), step)

        label_preds.append(output.flatten())
        label_trues.append(gt.flatten())
        count = count+1

    miou, ious = compute_mIoU(label_preds, label_trues, args.num_classes-1,
                              os.path.join(args.result_path,
                                           'preds_result_for_{}_iter.txt'.format(step)))
    mean_F1 = Mean_F1/count
    class_F1 = Class_F1/count
    simple_MF1 = np.mean(class_F1,axis=0)
    
    with open(os.path.join(args.result_path,'preds_result_for_{}_iter.txt'.format(step)), 'a+') as f:
        f.write('mean_F1: ' + str(mean_F1) + '\n')
        f.write('class_F1: ' + str(class_F1) + '\n')
        f.write('simple_MF1: ' + str(simple_MF1) + '\n')

    print('===>mIoU: {}'.format(miou))
    return miou, ious,mean_F1,class_F1,simple_MF1

def mytest2(model, step, testloader,device,writer,palette):
    model.eval()
    #interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
    data_list = []
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print ('The''iter of ', index, 'processed')
            image, label, size, name = batch
            size = size[0].numpy()
            _,output = model(Variable(image, volatile=True).cuda())
            #output = interp(output).cpu().data[0].numpy()

            
            upsample = nn.Upsample(size=(512,512),mode='bilinear', align_corners=True)
            output = upsample(output).cpu().data[0].numpy()
            
            output = output[:, :size[0], :size[1]]
            gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.uint8)

            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            gt = gt.flatten()
            output = output.flatten()
            class_F1 = f1_score(gt, output, average=None)
            mean_F1 = f1_score(gt, output, average='weighted')


            pred_img = colorize_mask(output,palette)
            name = name[0].strip()

            if index < 300 :
                pred_img.save('%s/%s' % (args.pred_path, name+'.tif'))

            if index % 50 == 0:
                # add image to tensorboard
                writer.add_image('diff/' + name[0],
                                 ToTensor()(pred_img.convert('RGB')), step)


            # if index % 100 == 0 and index !=0:
            #    show_all(gt, output,name[0])
            data_list.append([gt.flatten(), output.flatten()])
            #if index == 20:
            #    break

    miou,class_iou= get_iou(data_list, args.num_classes-1,
                              os.path.join(args.result_path,
                                           'preds_result_for_{}_iter.txt'.format(step)))

    print('===>mIoU: {}'.format(miou))

    return miou,class_iou,mean_F1,class_F1

def main():
    """Create the model and start the training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    # get 'palette' and 'label' from target dataset


    # Create network.
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    
    config_multi_trans = CONFIGS_ViT_seg[args.multi_trans_name]
    config_multi_trans.n_classes = args.num_classes
    config_multi_trans.n_skip = args.n_skip
    
    model = ViT_seg(config_vit, config_multi_trans, img_size=args.img_size, num_classes=config_vit.n_classes, multi_trans_depth = args.multi_trans_depth).cuda()


    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.pred_path):
        os.makedirs(args.pred_path)
    # model.load_state_dict(torch.load('../../datasets/MS_DeepLab_resnet_pretrained_COCO_init.pth'))


    # model.load_state_dict(torch.load(args.restore_from))
    # model_src.load_state_dict(torch.load(args.restore_from))

    # model.float()
    # model.eval() # use_global_stats = True
    checkpoint = 8000
    pretrained_dict = torch.load(args.restore_from)
    print("---start load pretrained modle---")
    model.load_state_dict(pretrained_dict)
    palette = [[255, 255, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 255, 255], [255, 0, 0]]
    labelnames = ['car', 'building', 'tree', 'veg', 'road']
    model.cuda()


    cudnn.benchmark = True

    target_testloader = data.DataLoader(
        ISPRS_POT_IRRG_TwoImage('fine', 'val', mean=IMG_MEAN_ISPRS),
        batch_size=1, shuffle=False, pin_memory=True
    )

    writer = SummaryWriter(log_dir=args.log_path)

    miou, ious,mean_F1,class_F1,simple_MF1 = mytest(model, checkpoint, target_testloader, device, writer,palette)

    print('meanIoU:', miou)
    print('class iou:',ious)
    print('mean_F1:', mean_F1)
    print('class_F1:', class_F1)
    print('simple_MF1:', simple_MF1)


if __name__ == '__main__':
    main()
