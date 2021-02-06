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
