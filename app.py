import cv2
import numpy as np
import argparse


def draw_contour(file_path):
    image = cv2.imread(file_path)

    filterd_image = cv2.medianBlur(image, 7)
    img_grey = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2GRAY)

    thresh = 100

    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = np.uint8(np.zeros((image.shape[0], image.shape[1])))

    cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)

    cv2.imshow('origin', image)
    cv2.imshow('res', img_contours)

    cv2.waitKey()
    cv2.destroyAllWindows()


def args_pars():
    parser = argparse.ArgumentParser(description='draw_contour args')
    parser.add_argument(
        '--file',
        type=str,
        default="./<path>",
        help='Путь к файлу'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = args_pars()
    draw_contour(args.file)