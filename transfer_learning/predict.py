import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


target_size = (229, 229)  # fixed size for InceptionV3 architecture


def predict(model, img, target_size):
    """Run model prediction on image
    Args:
        model: keras model
        img: PIL format image
        target_size: (w,h) tuple
    Returns:
        list of predicted labels and their probabilities
    """
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_prediction(dir_path, model_path, result_path):
    """
    :param dir_path: 所需要识别的图片目录
    :param model_path: 模型存放地址
    :param result_path: 输出的图片地址
    :return:
    """
    model = load_model(model_path)

    makedir(result_path)
    makedir(os.path.join(result_path, '0'))
    makedir(os.path.join(result_path, '1'))

    list = ['.jpg', '.jpeg', '.png']

    for file in os.listdir(dir_path):
        name = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1]
        if ext in list:
            img = Image.open(os.path.join(dir_path, file))
            if img.size != target_size:
                img = img.resize(target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)

            if preds[0][0] > preds[0][1]:
                predict_result = preds[0][0]
                label = '0'
            else:
                predict_result = preds[0][1]
                label = '1'

            # plt.imsave(os.path.join(result_path, label, "{}_{}_".format(label, "%.3f" % predict_result) + file), img)
            # plt.imsave(os.path.join(result_path, "{}\\{}_{}{}".format(label, name, "%.3f" % predict_result, ext)), img)
            img.save(os.path.join(result_path, "{}\\{}_{}{}".format(label, name, "%.3f" % predict_result, ext)))


def plot_preds(image, preds):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
        image: PIL image
        preds: list of predicted labels and their probabilities
    """
    plt.imshow(image)
    plt.axis('off')

    plt.figure()
    labels = ("cat", "dog")
    plt.barh([0, 1], preds, alpha=0.5)
    plt.yticks([0, 1], labels)
    plt.xlabel('Probability')
    plt.xlim(0, 1.01)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    # a.add_argument("--image", help="path to image", default=r"F:\DL\data\overlap\validate\Background\196(9).jpg")
    a.add_argument("--image", help="path to image", default=r"F:\DL\data\cord\test1\56.jpg")
    a.add_argument("--image_url", help="url to image")
    a.add_argument("--test_folder", help="path to test folder", default="F:\\DL\\data\\cord\\test1\\")
    a.add_argument("--result_folder", help="path to result folder", default="F:\\DL\\data\\cord\\result\\")
    a.add_argument("--model", default=r"C:\Users\gaohang\Desktop\DeepLearningSandbox\transfer_learning\inceptionv3-ft.model")
    args = a.parse_args()

    if args.image is None and args.image_url is None:
        a.print_help()
        sys.exit(1)

    if args.test_folder is not None and args.result_folder is not None:
        make_prediction(args.test_folder, args.model, args.result_folder)

    model = load_model(args.model)
    if args.image is not None:
        img = Image.open(args.image)
        preds = predict(model, img, target_size)
        plot_preds(img, preds)

    if args.image_url is not None:
        response = requests.get(args.image_url)
        img = Image.open(BytesIO(response.content))
        preds = predict(model, img, target_size)
        plot_preds(img, preds)
