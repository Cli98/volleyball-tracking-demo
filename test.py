import math
import argparse
import numpy as np
import random
import cv2
from network.network import model
import torch
import torch.backends.cudnn as cudnn
import os
from tracking_utils import generate_crop
from Visual_utils import draw_ball_path


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch volleyball project test script')
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
    parser.add_argument('--epochs', default=10, type=int,
                        help='How many epochs do you want to train your model')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='provide learning rate here to train your model')
    parser.add_argument('--print_step', default=40, type=int,
                        help='provide print step to print training stats')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # list all required variable below
    args = arg_parser()
    seed = args.seed
    root = args.root
    video_path = os.path.join(root, "data", "graz-arbesbach_3.avi")
    checkpoint_path = os.path.join(root, "checkpoint", "model_epoch_29.pth.tar")
    lr = args.lr
    scale = (args.resize_width, args.resize_height)

    # Setup seed for reproduction purpose
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # At the start we setup model for inference purpose
    # Since we inference image frame by frame, no dataloader is required here.
    # We send all samples to gpu id 0 for inference, if enabled
    inference_model = model(3, 32, 64, 32, 2)
    opt = torch.optim.SGD(inference_model.parameters(), lr=lr)
    cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss()
    inference_model.eval()
    if checkpoint_path:
        if os.path.isfile(checkpoint_path):
            # reload model weights and optimizer weights
            print("Reloading checkpoint at location: {}".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            state_dict = {k.replace("module.",""):checkpoint['state_dict'][k] for k in checkpoint['state_dict']}
            inference_model.load_state_dict(state_dict)
            opt.load_state_dict(checkpoint['optimizer'])
        else:
            print("The indicate checkpoint does not exist! please verify path!")
    else:
        print("No checkpoint loaded!")
        
    if torch.cuda.is_available() and args.gpu_num>0:
        gpu_ids = [i for i in range(args.gpu_num)]
        inference_model.to(gpu_ids[0])
        inference_model = torch.nn.DataParallel(inference_model, gpu_ids)
        criterion = criterion.cuda(gpu_ids[0])
    else:
        print("GPU is not available or not specified. Run with CPU mode")

    # Conduct preprocessing below
    print("The path for inference video is: {}".format(video_path))
    vs = cv2.VideoCapture(video_path)
    print("The FPS of the video is: {}".format(vs.get(cv2.CAP_PROP_FPS)))
    backSub = cv2.createBackgroundSubtractorMOG2()
    frame_count = 0
    if vs.isOpened():
        _, frame = vs.read()
    frame_count = 1
    if not os.path.exists(os.path.join(root, "frames")):
        os.makedirs(os.path.join(root, "frames"), exist_ok=False)
    bounding_box, prev_bounding_box, path_tracker = None, None, []

    while frame is not None:
        # preprocessing raw image
        mask = backSub.apply(frame)
        mask = cv2.dilate(mask, None)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # TODO: after complete processing, update cnt here
        path_tracker, prev_bounding_box, bounding_box, frame_count = generate_crop(mask, frame, scale, inference_model,
                                                            frame_count, bounding_box, prev_bounding_box, path_tracker)
        frame = draw_ball_path(frame, bounding_box, prev_bounding_box)

        # Generate tracking result
        cv2.imwrite("frames/frame-{:03d}.jpg".format(frame_count), frame)
        #cv2.imshow('frame', pic)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        _, frame = vs.read()
    