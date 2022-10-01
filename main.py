# =============================================================================
# Import required libraries
# =============================================================================
import argparse
import numpy as np
import torch
from torch import nn

from dataset import make_data_loader
from image_show import predicted_batch_plot
from models import *
from loss_functions import *
from engine import Engine


# =============================================================================
# Define hyperparameters
# =============================================================================
parser = argparse.ArgumentParser(
    description='PyTorch Training for Automatic Image Annotation')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training')
parser.add_argument('--data_root_dir', default='./Corel-5k/', type=str)
parser.add_argument('--image-size', default=448, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--num_workers', default=2, type=int,
                    help='number of data loading workers (default: 2)')
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--loss-function', metavar='NAME',
                    help='loss function (e.g. BCELoss)')
parser.add_argument('--model', metavar='NAME',
                    help='model name (e.g. ResNeXt50)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluation of the model on the validation set')
parser.add_argument('--augmentation', dest='augmentation', action='store_true',
                    help='using more data for training')
parser.add_argument(
    '--save_dir', default='./checkpoints/', type=str, help='save path')


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    is_train = True if not args.evaluate else False

    train_loader, validation_loader, classes = make_data_loader(args)
    
    if args.model == 'ResNet101':
        model = ResNet101(args, len(classes), pretrained=is_train)
    elif args.model == 'ResNeXt50':
        model = ResNeXt50(args, len(classes), pretrained=is_train)
    elif args.model == 'Xception':
        model = Xception(args, len(classes), pretrained=is_train)
    elif args.model == 'TResNet':
        model = TResNet(args, len(classes), pretrained=is_train)

    if args.loss_function == 'BCELoss':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif args.loss_function == 'FocalLoss':
        criterion = MultiLabelLoss(gamma_neg=3,
                                   gamma_pos=3,
                                   neg_margin=0)
    elif args.loss_function == 'AsymmetricLoss':
        criterion = MultiLabelLoss(gamma_neg=4,
                                   gamma_pos=0,
                                   neg_margin=0.05)
    elif args.loss_function == 'LSEPLoss':
        criterion = LSEPLoss()

    engine = Engine(args,
                    model,
                    criterion,
                    train_loader,
                    validation_loader,
                    len(classes))

    if is_train:
            engine.initialization(is_train)
            engine.train_iteration()
    else:
        engine.initialization(is_train)
        engine.load_model()
        print('Computing best thresholds: ')
        best_thresholds = engine.matthew_corrcoef(train_loader)
        print(best_thresholds)
        engine.validation(dataloader=validation_loader,
                          mcc=True,
                          thresholds=best_thresholds)
        # show images and predicted labels
        images, annotations = iter(validation_loader).next()
        if engine.train_on_GPU():
            images = images.cuda()
        predicted_batch_plot(args,
                             classes,
                             model,
                             images,
                             annotations,
                             best_thresholds=None)
        #
        predicted_batch_plot(args,
                             classes,
                             model,
                             images,
                             annotations,
                             best_thresholds=best_thresholds)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)