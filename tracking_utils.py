import cv2
import torch
from volleyball import volleyball
import numpy as np


def process_volleyball(path_tracker, bounding_box, prev_bounding_box, frame_count, x, y, r):
    bounding_box = process_element(path_tracker, frame_count, x, y)
    if not bounding_box:
        # No element in current tracking task
        path_tracker.append(volleyball([x, y, r, frame_count]))
        return [path_tracker, None, None]
    bounding_box.add([x, y, r, frame_count])
    if bounding_box.status == 2:
        if not prev_bounding_box or len(bounding_box.coord) > len(prev_bounding_box.coord):
            prev_bounding_box = bounding_box
    return [path_tracker, prev_bounding_box, bounding_box]


def process_element(path_tracker, frame_count, x, y):
    static_array, redicted_array = [], []
    for vball in path_tracker:
        status, distance = vball.fit(-1, x, y)
        if status:
            if frame_count - vball.frame_count<=10:
                redicted_array.append([vball, distance])
            elif vball.status == 1:
                static_array.append([vball, distance])
    if len(static_array) == len(redicted_array) == 0:
        return
    redicted_array.sort(key= lambda x:x[1])
    if redicted_array:
        return redicted_array[0][0]
    static_array.sort(key= lambda x:x[1])
    if static_array:
        return static_array[0][0]



def generate_crop(mask, frame, scale, model, cnt, bounding_box, prev_bounding_box, path_tracker):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Here bounding_box is a collection of points for tracking purpose
    # print(type(contours), len(contours))
    for contour in contours:
        # Check if initial crops suitable, the next check will guess if the crops are targets.
        #print(type(contour), contour.shape)
        rx, ry, rw, rh = cv2.boundingRect(contour)
        mn = min(rw, rh)
        mx = max(rw, rh)
        r = mx / mn
        #print(cnt, mn, mx, r)
        if mn < 10 or mx > 40 or r > 1.5:
            continue

        cut_m = mask[ry: ry + rh, rx: rx + rw]
        # coord of cut_m: 0, 0, rh, rw
        # Check the mask and remove background crops as much as possible
        # instead of picking highest
        dy = 2.0/5*rh
        dx = 2.0/5*rw

        m_sub_x = cut_m[:int(rh+dy), :]
        m_sub_y = cut_m[: ,:int(rw+dx)]
        pixel_x = cv2.countNonZero(m_sub_x)
        pixel_y = cv2.countNonZero(m_sub_y)
        pixel_xy = cv2.countNonZero(cut_m)
        if not(pixel_x/pixel_xy>0.15 and pixel_y/pixel_xy>0.15) or pixel_xy/(rh*rw)<0.5:
            continue
        # Once pass this point, we treat current crops as potenial region.
        # This is not the same as before where we treat only the highest point as candidate
        cut_f = frame[ry: ry + rh, rx: rx + rw]
        cut_c = cv2.bitwise_and(cut_f, cut_f, mask=cut_m)
        cut_c = cv2.resize(cut_c, scale, interpolation = cv2.INTER_CUBIC)
        cut_c = torch.from_numpy(np.transpose(cut_c, [2,0,1])).float()
        output = model(cut_c.unsqueeze(0), 1)
        pred = torch.argmax(output, dim=1)
        if pred == 1:
            # if we detect a ball, wrap up with bbox/circle
            ((x, y), r) = cv2.minEnclosingCircle(contour)
            [path_tracker, prev_bounding_box, bounding_box] = \
                process_volleyball(path_tracker, bounding_box, prev_bounding_box, cnt, x, y, r)
            print("Ball detected for frame {}".format(cnt))

    cnt += 1
    print("{} complete!".format(cnt))
    return path_tracker, prev_bounding_box, bounding_box, cnt