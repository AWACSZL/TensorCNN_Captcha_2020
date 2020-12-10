from captcha.image import ImageCaptcha
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import datetime

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
SAVE_PATH = ".\\model"
CHAR_SET = number+alphabet+ALPHABET
CHAR_SET_LEN = len(CHAR_SET)
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160


def random_captcha_text(char_set=None, captcha_size=4):
    if char_set is None:
        char_set = number + alphabet + ALPHABET

    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image(width=160, height=60, char_set=CHAR_SET):
    image = ImageCaptcha(width=width, height=height)

    captcha_text = random_captcha_text(char_set)
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


text, image = gen_captcha_text_and_image(char_set=CHAR_SET)
MAX_CAPTCHA = len(text)
print('CHAR_SET_LEN=', CHAR_SET_LEN, ' MAX_CAPTCHA=', MAX_CAPTCHA)


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def text2vec(text):
    vector = np.zeros([MAX_CAPTCHA, CHAR_SET_LEN])
    for i, c in enumerate(text):
        idx = CHAR_SET.index(c)
        vector[i][idx] = 1.0
    return vector


def vec2text(vec):
    text = []
    for i, c in enumerate(vec):
        text.append(CHAR_SET[c])
    return "".join(text)


def get_next_batch(batch_size=128): #每一批次包含128张图像
    #batch_x为numpy数值化候的图像，batch_y为该图像的字符标签
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA, CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while True:
            #随机生成验证码，与前面生成验证码部分相同
            text, image = gen_captcha_text_and_image(char_set=CHAR_SET)
            #规定验证码尺寸：高60，宽160，像素通道3，即为彩色
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = tf.reshape(convert2gray(image), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))#二值化与重塑图像尺寸
        batch_x[i, :] = image
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def crack_captcha_cnn():
    model = tf.keras.Sequential()#实例化一个Sequential类，该类是继承于tensorflow内部的Model类；

    model.add(tf.keras.layers.Conv2D(32, (3, 3)))#具有32个卷积核的卷积层，过滤器大小3*3
    #普通二维卷积，常用于图像。参数个数 = 输入通道数×卷积核尺寸(如3乘3)×卷积核个数
    model.add(tf.keras.layers.PReLU()) #参数矫正线性单元，继承自tensorflow的层，与轴形状有关
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2)) #2D图像池化层，池大小2*2，步长值为2

    model.add(tf.keras.layers.Conv2D(64, (5, 5)))#卷积核与过滤器持续增大，
    model.add(tf.keras.layers.PReLU())#同上
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))#同上

    model.add(tf.keras.layers.Conv2D(128, (5, 5)))#卷积核与过滤器持续增大，
    model.add(tf.keras.layers.PReLU())#同上
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))#同上

    model.add(tf.keras.layers.Flatten())  #压平层，用于将多维张量压成一维。即降维
    model.add(tf.keras.layers.Dense(MAX_CAPTCHA * CHAR_SET_LEN))
    #上面为密集连接层。参数个数 = 输入层特征数× 输出层特征数(weight)＋ 输出层特征数(bias)
    model.add(tf.keras.layers.Reshape([MAX_CAPTCHA, CHAR_SET_LEN]))
    #上面为形状重塑层，改变输入张量的形状。
    model.add(tf.keras.layers.Softmax())#激活函数层，采用Softmax激活函数
    return model


def train():
    try:
        model = tf.keras.models.load_model(SAVE_PATH + 'model')#尝试载入上一个模型
    except Exception as e:
        print('#######Exception', e)
        model = crack_captcha_cnn()#如果没有模型（即第一次），就调用神经网络创建一个新的模型

    model.compile(optimizer='Adam', #优化器——adam
                  metrics=['accuracy'], #metrics: 列表，包含评估模型在训练和测试时的性能的指标，
                  loss='categorical_crossentropy')  #交叉熵损失函数

    for times in range(500000):  #设定训练次数500000次
        batch_x, batch_y = get_next_batch(512)  #传入上述的batch批次类，一共为128*4=512张
        print('times=', times, ' batch_x.shape=', batch_x.shape, ' batch_y.shape=', batch_y.shape)
        model.fit(batch_x, batch_y, epochs=4) #把两项数据传入模型，使其开始运作，epoch为类似batch的批次
        print("y预测=\n", np.argmax(model.predict(batch_x), axis=2))#模型对图像进行预测，沿着第2轴展开
        print("y实际=\n", np.argmax(batch_y, axis=2))#输出图像实际标签的值

        if 0 == times % 10:
            print("save model at times=", times)
            model.save(SAVE_PATH + '%d'%(times)) #根据训练次数保存模型


def predict():                        # ".\\model"
    model = tf.keras.models.load_model(SAVE_PATH + 'model')#导入模型
    success = 0 #初步定义成功率
    count = 100
    for _ in range(count):
        data_x, data_y = get_next_batch(1)
        prediction_value = model.predict(data_x)
        data_y = vec2text(np.argmax(data_y, axis=2)[0])
        prediction_value = vec2text(np.argmax(prediction_value, axis=2)[0])

        if data_y.upper() == prediction_value.upper():
            print("y预测=", prediction_value, "y实际=", data_y, "预测成功。")
            success += 1
        else:
            print("y预测=", prediction_value, "y实际=", data_y, "预测失败。")
        print("预测", count, "次", "成功率=", success / count)

    pass


if __name__ == "__main__":
    train()
    predict()