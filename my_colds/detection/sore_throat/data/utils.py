import cv2


def load_img_from_disk(path):
    return cv2.imread(path, cv2.COLOR_BGR2RGB)


def save_img(path, img):
    cv2.imwrite(path, img)


def load_img_via_http(link):
    raise NotImplementedError