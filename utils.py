import numpy as np
from PIL import Image


# 生成更多数据使用
def data_augmentation(image, mode):  # image的类型是np.array
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


# np.array() 是 NumPy 库中的一个函数，用于创建一个 NumPy 数组。NumPy 数组是一个高效的多维数组对象，用于存储和操作大量数据，特别适用于数值计算、矩阵操作等任务。
def load_images(file):
    im = Image.open(file)  # im是image类的的对象
    return np.array(im, dtype="float32") / 255.0  # 归一化


def save_images(filepath, result_1, result_2=None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis=1)   # axis=1表示图片左右拼接

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))   # 解归一化，以uint8保存
    im.save(filepath, 'png')     # 保存图像