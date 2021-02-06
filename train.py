import torch
from network.network import model
from dataloader.dataloader import Volleyball_loader
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
import os

#TODO: Test your dataloader before you run it
#TODO: adjust learning rate


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch volleyball project')
    parser.add_argument('root', metavar='DIR', default="", type=str,
                        help='path to dataset')
    parser.add_argument('--affix', default="jpg", type=str,
                        help='image affix')
    parser.add_argument('--resume', type=str,
                        help='resume process of model training')
    parser.add_argument('--checkpoint_path', default="./checkpoint", type=str,
                        help='where to save model checkpoint')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='The batchsize to train classifier')
    parser.add_argument('--gpu_num', default=0, type=int,
                        help='How many gpu to train model? set 0 to disable gpu training')
    parser.add_argument('--seed', default=2020, type=int,
                        help='Set a seed for reproduction purpose')
    parser.add_argument('--resize_height', default=64, type=int,
                        help='provide height here if you need to resize')
    parser.add_argument('--resize_width', default=64, type=int,
                        help='provide width here if you need to resize')
    parser.add_argument('--split_ratio', default=0.1, type=float,
                        help='provide split ratio for train/test split')
    parser.add_argument('--epoch', default=10, type=int,
                        help='How many epochs do you want to train your model')
    parser.add_argument('--', default=0.01, type=float,
                        help='provide learning rate here to train your model')
    parser.add_argument('--print_step', default=40, type=int,
                        help='provide print step to print training stats')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # model required parameters start here
    args = arg_parser()
    scale = (args.resize_width, args.resize_height)

    # Setup seed for reproduction purpose
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # General setup
    gpu_ids = []
    train_dataset = Volleyball_loader(args.root, args.affix, scale=scale, mode="train", split_ratio=args.split_ratio)
    train_dataloader = torch.utils.data.Dataloader(train_dataset, batch_size = args.batch_size, shuffle = True)
    val_dataset = Volleyball_loader(args.root, args.affix, scale=scale, mode="val", split_ratio=args.split_ratio)
    val_dataloader = torch.utils.data.Dataloader(val_dataset, batch_size = 1, shuffle = True)
    print("Total number of training samples are {} and validation samples are {}".format(len(train_dataloader),
                                                                                         len(val_dataloader)))

    model = model()
    opt = torch.optim.sgd(model.parameters, lr = 0.01)
    criterion = torch.nn.CrossEntropyLoss().cuda(gpu_ids[0])
    cudnn.benchmark = True

    if args.resume:
        if os.path.isfile(args.resume):
            print("Reloading checkpoint: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            opt.load(checkpoint['optimizer'])
        else:
            print("The requested checkpoint is not available!")

    if torch.cuda.is_available() and args.gpu_num>0:
        gpu_ids = [i for i in range(args.gpu_num)]
        model.to(gpu_ids[0])
        model = torch.nn.DataParallel(model, gpu_ids)
    else:
        print("GPU is not available or not specified. Run with CPU mode")

    current_best_acc = float("-inf")
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path, exist_ok=False)


    for epoch in range(args.epochs):
        train_epoch_loss, train_epoch_acc = 0, 0
        # Switch to train mode
        model.train()
        # setup lr decay
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        for i, (pic, target) in enumerate(train_dataloader):
            if args.gpu_num>0:
                pic = pic.cuda(gpu_ids[0])
                target = target.cuda(gpu_ids[0])
            output = model(pic)
            pred = torch.argmax(output, dim=1)

            loss = criterion(output, target)
            train_epoch_loss += loss
            acc = pred.eq(target).sum().float().item()/args.batch_size
            train_epoch_acc += acc

            if i>0 and i%args.print_step==0:
                print("Step {} loss : {:.2f} acc : {:.2f}".format(i, loss, acc))

            opt.zero_grad()
            loss.backward()
            opt.step()

        val_epoch_loss, val_epoch_acc = 0, 0
        model.eval()
        for i, (pic, target) in enumerate(val_dataloader):
            if args.gpu_num>0:
                pic = pic.cuda(gpu_ids[0])
                target = target.cuda(gpu_ids[0])
            output = model(pic)
            pred = torch.argmax(output, dim=1)

            loss = criterion(output, target)
            val_epoch_loss += loss
            acc = pred.eq(target).sum().float().item()/args.batch_size
            val_epoch_acc += acc
        if current_best_acc<val_epoch_acc:
            save_dict = {
                "state_dict":model.state_dict(),
                "optimizer":opt.state_dict()
            }
            torch.save(os.path.join(args.checkpoint_path))
        print("validate epoch {} loss : {:.2f} , acc : {:.2f}".format(epoch, val_epoch_loss, val_epoch_acc))



