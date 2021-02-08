import math
import numpy as np
import random
import cv2
from network.network import model
import torch
import torch.backends.cudnn as cudnn
import os
from tracking_utils import generate_crop
from Visual_utils import draw_ball_path


#TODO: add args for test.py
if __name__ == "__main__":
    # list all required variable below
    args = None
    seed = 2020
    root = ""
    video_path = ""
    checkpoint_path = ""
    lr = 0.01
    scale = (args.resize_width, args.resize_height)

    # Setup seed for reproduction purpose
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # At the start we setup model for inference purpose
    # Since we inference image frame by frame, no dataloader is required here.
    # We send all samples to gpu id 0 for inference, if enabled
    inference_model = model(3, 32, 64, 32, 2, args.batch_size)
    opt = torch.optim.SGD(inference_model.parameters(), lr=lr)
    cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss()
    inference_model.eval()
    if checkpoint_path:
        if os.path.isfile(checkpoint_path):
            # reload model weights and optimizer weights
            print("Reloading checkpoint at location: {}".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
        else:
            print("The indicate checkpoint does not exist! please verify path!")
    else:
        print("No checkpoint loaded!")
        
    if torch.cuda.is_available() and args.gpu_num>0:
        gpu_ids = [i for i in range(args.gpu_num)]
        model.to(gpu_ids[0])
        model = torch.nn.DataParallel(model, gpu_ids)
        criterion = criterion.cuda(gpu_ids[0])
    else:
        print("GPU is not available or not specified. Run with CPU mode")

    # Conduct preprocessing below
    vs = cv2.VideoCapture(video_path)
    backSub = cv2.createBackgroundSubtractorMOG2()
    frame_count = 0
    if vs.isOpened():
        _, frame = vs.read()
    frame_count = 1

    while frame:
        # preprocessing raw image
        mask = backSub.apply(frame)
        mask = cv2.dilate(mask, None)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # TODO: after complete processing, update cnt here
        path_tracker, prev_bounding_box, bounding_box, frame_count = generate_crop()
        pic = draw_ball_path(frame, bounding_box, prev_bounding_box)

        # Generate tracking result
        cv2.imwrite("frames/frame-{:03d}.jpg".format(frame_count), pic)
        cv2.imshow('frame', pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        frame_count += 1
    