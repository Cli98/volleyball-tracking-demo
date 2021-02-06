import cv2
from data_preprocesssing_utils import check_dir
import os

"""
Objective: Use strong assumption to manually guess the location of 
volleyball. This can help save time to do manual labelling.
Not very effective but works.
"""


def guess_volleyball_pos(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    chigh = None
    cy = 10000
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        s = min(w, h)
        if s < 15:
            continue
        r = max(w, h) / s
        if r > 1.5:
            continue
        if y < cy:
            # Try to find the highest point
            cy = y
            chigh = c
    return chigh


def locate_volleyball_pos(clip_path, mask_out_path=None, pic_out_path=None, verbose = 0):
    """
    Only support single video by now
    :param clip_path:
    :param mask_out_path:
    :param pic_out_path:
    :return:
    """
    check_dir(mask_out_path)
    check_dir(pic_out_path)
    vs = cv2.VideoCapture(clip_path)
    backSub = cv2.createBackgroundSubtractorMOG2()  # Extract object mask
    n = 0
    if vs.isOpened():
        frame = True

    while frame is not None:
        _, frame = vs.read()

        mask = backSub.apply(frame)
        if verbose:
            cv2.imshow("raw mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        if verbose:
            cv2.imshow("blur mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if verbose:
            cv2.imshow("threshold mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        chigh = guess_volleyball_pos(mask)
        if not (chigh is None):
            rx, ry, rw, rh = cv2.boundingRect(chigh)
            cut_mask = mask[ry: ry + rh, rx: rx + rw]
            if verbose:
                cv2.imshow("original image", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if mask_out_path:
                cv2.imwrite("{0}/b-{1:03d}.jpg".format(mask_out_path, n), cut_mask)
                if verbose:
                    cv2.imshow("output mask", mask)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            if pic_out_path:
                cut_frame = frame[ry: ry + rh, rx: rx + rw]
                process_img = cv2.bitwise_and(cut_frame, cut_frame, mask=cut_mask)
                cv2.imwrite("{0}/c-{1:03d}.jpg".format(pic_out_path, n), process_img)
                if verbose:
                    cv2.imshow("output frame", cut_frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    cv2.imshow("output picture", process_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        print("{} frames have been processed! ".format(n))
        n += 1


if __name__ == "__main__":
    video_name = "graz-klagenfurt1_2.avi"
    clip_path = os.path.join("data", video_name)
    video_prefix, video_affix = video_name.split(".")
    mask_out_path = os.path.join("data", "mask", video_prefix)
    pic_out_path = os.path.join("data", "image", video_prefix)
    check_dir(mask_out_path)
    check_dir(pic_out_path)
    verbose = 0
    locate_volleyball_pos(clip_path, mask_out_path=mask_out_path, pic_out_path=pic_out_path, verbose = verbose)
