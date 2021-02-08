import cv2

def draw_high_cont(path, chigh, save_path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # chigh = guess_volleyball_pos(mask)
    if chigh is not None:
        cmask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cmask, [chigh], 0, (255, 0, 0), 2)
        cv2.imwrite(save_path, cmask)
        cv2.imshow('frame', cmask)
        cv2.waitKey()


def draw_ball_path(frame, bounding_box, prev_bounding_box):
    if bounding_box:
        cv2.circle(frame, (bounding_box[-1][0], bounding_box[-1][1]), 10, (0, 200, 0), 3)
        for point in bounding_box.coord:
            cv2.circle(frame, (point[0], point[1]), 3, (150, 150, 150), -1)
    elif prev_bounding_box:
            x, y = prev_bounding_box.predict()
            cv2.circle(frame, (x, y), 10, (0, 200, 0), 3)
    return frame

