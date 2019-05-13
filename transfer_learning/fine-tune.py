import os

import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from keras.callbacks import ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"


# 数据准备
# 默认值
IM_WIDTH, IM_HEIGHT = 229, 229  # fixed size for InceptionV3
NB_EPOCHS = 3  # 转移学习迭代次数
BAT_SIZE = 32  # batch size 每次喂样本数
FC_SIZE = 1024  # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 172  # 冻结层的数量


# 统计样本数量
def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


# 冻上base_model所有层，这样就可以正确获得bottleneck特征
def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """
    Add last layer to the convnet 添加全连接层

    Args:
        base_model: keras model excluding top
        nb_classes: # of classes

    Returns:
        new keras model with last layer
    """
    # base_model.summary()
    # x = base_model.layers[279].output
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    # x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    # model.summary()
    return model


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

    Args:
        model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_train_samples = get_nb_files(args.train_dir)  # 训练样本个数
    nb_classes = len(glob.glob(args.train_dir + "/*"))  # 分类数
    nb_val_samples = get_nb_files(args.val_dir)  # 验证集样本个数
    nb_epoch = int(args.nb_epoch)  # epoch数量
    batch_size = int(args.batch_size)

    # data prep # 图片生成器
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # 应用于每个输入的函数。在任何其他改变之前运行。需要一个参数：一张图像（秩为3的Numpy张量），并且应该输出一个同尺寸的 Numpy 张量。
        rotation_range=30,  # 整数。随机旋转的度数范围。
        width_shift_range=0.2,  # 浮点数、一维数组或整数 float: 如果 <1，则是除以总宽度的值
        height_shift_range=0.2,  # 浮点数、一维数组或整数 float: 如果 <1，则是除以总高度的值，
        shear_range=0.2,  # 浮点数。剪切强度（以弧度逆时针方向剪切角度）。
        zoom_range=0.2,  # 浮点数 或 [lower, upper]。随机缩放范围。如果是浮点数，[lower, upper] = [1-zoom_range, 1+zoom_range]。
        horizontal_flip=True,  # 布尔值。随机水平翻转。
        # rescale=1/255.0  # 重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）。
        # samplewise_center = True, # 布尔值。将每个样本的均值设置为 0。
        # samplewise_std_normalization = True # 布尔值。将每个输入除以其标准差。
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        # rescale=1/255.0
        # samplewise_center = True,
        # samplewise_std_normalization = True
    )

    # 训练数据与测试数据
    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    validation_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    # setup model 预先要下载no_top模型
    base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
    # for i, layer in enumerate(base_model.layers):  # 打印各卷积层的名字
    #    print(i, layer.name)
    model = add_new_last_layer(base_model, nb_classes)              # 从基本no_top模型上添加新层

    # transfer learning
    setup_to_transfer_learn(model, base_model)

    # 模型的保存目录
    save_dir = os.path.join(args.output_model_dir, 'saved_models_' + str(IM_WIDTH) + '_' + str(IM_HEIGHT))
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    filepath = "InceptionV3_TL_{epoch:02d}_{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc', verbose=1, save_best_only=True)
    history_tl = model.fit_generator(
        generator=train_generator,
        epochs=nb_epoch,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        class_weight='auto',
        verbose=1,
        callbacks=[checkpoint])

    # fine-tuning
    setup_to_finetune(model)

    filepath = "InceptionV3_FT_{epoch:02d}_{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc', verbose=1, save_best_only=True)
    history_ft = model.fit_generator(
        generator=train_generator,
        epochs=nb_epoch,
        steps_per_epoch=nb_train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        class_weight='auto',
        verbose=1,
        callbacks=[checkpoint])

    # 最终模型的保存
    model.save(args.output_model_file)

    if args.plot:
        plot_training(history_ft)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default=r"F:\DL\data\cord\catordog\train")  # 训练集数据
    a.add_argument("--val_dir", default=r"F:\DL\data\cord\catordog\validate")  # 验证集数据
    a.add_argument("--output_model_dir", default=r"F:\DL\data\cord\catordog\model")  # 上次最好结果模型的全路径名
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_file", default="inceptionv3-ft-catordog.model")
    a.add_argument("--plot", action="store_true", default=True)

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    train(args)
