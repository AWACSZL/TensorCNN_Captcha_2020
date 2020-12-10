#1
import cv2
import numpy as np
import sys
#2
from captcha.image import ImageCaptcha
import sys
import random
#3
import sys
#import image_preprocess_get1
import os
import cv2
#4
import cv2
import glob
#5
from PIL import Image
import os.path
import glob
import cv2
#6
import tensorflow as tf
from PIL import Image
import numpy as np
#import image_packing_get4

#7
import socket# 客户端 发送一个数据，再接收一个数据



#1_image preprocess get 1=====# 图像处理展示

'''
此文件为图像处理文件，包括预处理，去噪，二值化，归一化等。
！！！此为第一步！
'''

class image_process:

    def __init__(self):
        pass

    def path(self):
        path = '.\\image\\0166.png'
        return path

    # opencv读取原图片===========================================================
    def img_input(self,img_open_path):  # 参数为图像路径
        img = cv2.imread(img_open_path, 1)  # cv读入图像路径的图像，1为读取全像素通道颜色
        print(img.shape)
        # cv2.imshow('image', img)
        # cv2.waitKey(5000)
        # 显示原始图片
        return img

    # 灰度与模糊处理===========================================================
    def gray_blurred(self,img):  # 参数为原图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cv读取原图像，采用cv灰度化算法
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)  # 将灰度化的图像进行模糊化,0为灰白像素通道

        cv2.imwrite('.\\dilateimage\\' + 'gray.png', gray)  # 写入进路径文件
        return gray, blurred  # 返回灰度化、模糊化图像

    # 模糊图像二值化===========================================================
    def getbinary(self,blurred):  # 参数为模糊化图像
        ret, binary = cv2.threshold(blurred, 216, 225, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 将模糊化的灰度图像进行二值化，216为判断阈值，225是更改像素，后面为cv2算法
        print("threshold value: %s" % ret)
        cv2.imwrite('.\\dilateimage\\' + 'binary.png', binary)
        return binary

    # 膨胀化===========================================================
    def getdilate(self,binary):
        kernel = np.ones((4, 4), dtype=np.uint8)  # 返回全是1的n维数组
        dilate = cv2.dilate(binary, kernel, 1)  # 1:迭代次数，也就是执行几次膨胀操作
        cv2.imwrite('.\\dilateimage\\' + 'dilate.png', dilate)
        return dilate


    # 提取图像梯度===========================================================
    def gradient(self,dilate):
        gradX = cv2.Sobel(dilate, ddepth=cv2.CV_32F, dx=1, dy=0)
        gradY = cv2.Sobel(dilate, ddepth=cv2.CV_32F, dx=0, dy=1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        return gradient


    # 垂直投影像素===========================================================
    def getVProjection(self,dilate):
        # 将image图像转为黑白二值图，ret接收当前的阈值，thresh1接收输出的二值图
        ret, thresh1 = cv2.threshold(dilate, 127, 255, cv2.THRESH_BINARY)
        (h, w) = thresh1.shape  # 返回高和宽
        a = [0 for z in range(0, w)]  # a = [0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数

        # 记录每一列的波峰
        for j in range(0, w):  # 遍历一列
            for i in range(0, h):  # 遍历一行
                if thresh1[i, j] == 0:  # 如果该点为黑点
                    a[j] += 1  # 该列的计数器加一计数
                    thresh1[i, j] = 255  # 记录完后将其变为白色
        for j in range(0, w):  # 遍历每一列
            for i in range((h - a[j]), h):  # 从该列应该变黑的最顶部的点开始向最底部涂黑
                thresh1[i, j] = 0  # 涂黑
        return thresh1



    # 判定图像分割位置===========================================================
    def image_cut(self,dilate):

        fw = open('.\\data\\location.txt', "w")  # 清空之前的txt内像素值
        fw.truncate()

        ret, thresh1 = cv2.threshold(dilate, 127, 255, cv2.THRESH_BINARY)
        (h, w) = dilate.shape  # 返回高和宽
        a = [0 for z in range(0, w)]  # a = [0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数'
        b = [0 for z in range(0, w)]  # a = [0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数'

        pixelnum = 0
        position = {'left': 0, 'right': 0, 'value': 0}
        location = {'left': 0, 'right': 0}

        print('---------以下分割为像素位置的输出---------')

        # 记录每一列的波峰
        for j in range(0, w):  # 遍历一列
            for i in range(0, h):  # 遍历一行
                pixelnum = 0
                if thresh1[i, j] == 0:  # 如果该点为黑点
                    a[j] += 1  # 该列的计数器加一计数
            if a[j] == 0:  # 如果该列像素点为黑点的数量为0，则判定为空列
                position['right'] = j  # 记录该空列的为右区间点
                location['right'] = j
                for k in range(position['left'], position['right']):  # 从该两个列中进行像素量计算  #遍历一列
                    for n in range(0, h):  # 遍历一行
                        if thresh1[n, k] == 0:  # 如果该点为黑点
                            b[k] += 1  # 该列的黑点数量+1
                            pixelnum = b[k] + pixelnum  # 计算列区间内所有像素点的总和
                            position['value'] = pixelnum
                if pixelnum <= 10:
                    continue
                # print(position['left'], position['right'])
                # print(pixelnum)
                print('----------------')
                print(position['left'], position['right'], position['value'])
                print(location['left'], position['right'])

                with open('.\\data\\location.txt', 'a') as log:
                    for value in location.values():
                        log.write('{} '.format(value))  # 输出数据+空格到txt
                # fw = open(".\\data\\location.txt", 'a+')  #打开txt文件
                # fw.write(str(position))   # 把字典转化为str并写入
                # fw.write('\n')   #换行
                # fw.close()         #关闭
            position['left'] = position['right']  # 转换左区间点
            location['left'] = location['right']

    # 裁剪图像==========================================================================
    def confirmpoint(self,dilate):

        # dilate=cv2.imread(.\\dilateimage\\+'dilate.png',0)

        print('------以下为分割图像的输出------')

        count = len(open(r".\\data\\location.txt", 'r').readlines())  # 读取txt文件中的数据有多少行
        print('原始行数:', count)

        # with open(".\\data\\location.txt", "r") as f:
        #    data = f.readlines()
        #    print(data)

        with open('.\\data\\location.txt') as jaf:
            listdata = jaf.readline()  # 读取数据行
            newlistdata = listdata.strip().split(' ')  # 将数据转换为字符列表
            print('字符型列表数据输出:', newlistdata)  # 输出字符列表

        intdata = [int(x) for x in newlistdata]  # 转换字符列表为整数列表
        print('整数型列表数据输出:', intdata)  # 输出整数列表

        i = -1
        # 拆分列表，将一个列表拆分成每组坐标为准的新列表
        n = 2  # 将列表每2个组成一个小列表，
        for j in range(0, len(intdata), n):
            i = i + 1
            x, y = intdata[j:j + n]
            def plti(im, i):
                """
                画图的辅助函数
                """
                # im = im[:250, 50:100, :]
                # plt.imshow(im, interpolation="none", **kwargs)
                # plt.axis('off') # 去掉坐标轴
                # plt.subplot(), plt.imshow(im), plt.title('')
                # plt.savefig(".\\cut_image\\%d.jpg" % (i + 1),transparent=True, dpi=300, pad_inches = 0)
                # plt.show()
            im = dilate[:, x - 5:y + 5, :]  # 直接切片对图像进行裁剪[底高：顶高，左宽：右宽，未知]
            gray1, blurred1 = self.gray_blurred(im)
            binary1 = self.getbinary(blurred1)
            dilate1 = self.getdilate(binary1)
            cv2.imwrite(".\\cut_image\\%d.jpg" % (i), binary1)
            # plti(dilate1,i)
            print('拆分整数列表:', x, y)
        return x, y

    # 展示预处理的整体图片========================================================================
    # def show_img(img,gray,blurred,binary,dilate,gradient,thresh1):
    #    plt.subplot(421), plt.imshow(img), plt.title("1")
    #    plt.subplot(422), plt.imshow(gray), plt.title("2")
    #    plt.subplot(423), plt.imshow(blurred), plt.title("3")
    #    plt.subplot(424), plt.imshow(binary), plt.title("4")
    #    plt.subplot(425), plt.imshow(dilate), plt.title("5")
    #    plt.subplot(426), plt.imshow(gradient), plt.title("6")
    #    plt.subplot(427), plt.imshow(thresh1), plt.title("7")
    #    plt.subplot(428), plt.imshow(dst), plt.title("8")
    #    plt.show()

    # 输出控制台到文件===========================================================================
    class Logger(object):
        def __init__(self, fileN="Default.log"):
            self.terminal = sys.stdout
            self.log = open(fileN, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.flush()  # 每次写入后刷新到文件中，防止程序意外结束

        def flush(self):
            self.log.flush()

    sys.stdout = Logger(".\\data\\log.txt")

    '''
    if __name__ == '__main__':
        img_path=path()
        img=img_input(img_path)
        gray,blurred=gray_blurred(img)
        binary=getbinary(blurred)
        dilate=getdilate(binary)
        gradient=gradient(dilate)
        thresh1=getVProjection(dilate)
        image_cut(dilate)
        dst=doubleget(img)
        #show_img(img,gray,blurred,binary,dilate,gradient,thresh1)
    '''


#2 create_captcha创建验证码================
'''
生成验证码，数据产生部分
'''
class create_captcha:
    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    Anumber = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U','V', 'W', 'X', 'Y', 'Z']
    anumber = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u','v', 'w', 'x', 'y', 'z']

    num = 50000

    def __init__(self):
        pass

    # 获取四个随机字符
    # def random_captcha_text(char_set=number,captcha_size=4):
    def random_captcha_text(self,char_set=number, captcha_size=4):
        # char_set为预定的字符数组，captcha_size为字符数量
        captcha_text = []  # 验证码列表
        for i in range(captcha_size):
            # 随机选择
            c = random.choice(char_set)
            # 加入验证码列表
            captcha_text.append(c)

        return captcha_text

    # 生成字符对应的验证码
    def gen_captcha_text_and_image(self):
        image = ImageCaptcha()
        # 获取随机生成的验证码
        captcha_text = self.random_captcha_text()
        # 把验证码列表转换为字符串
        captcha_text = ''.join(captcha_text)
        # 生成验证码
        image.write(captcha_text, './image/' + captcha_text + '.png')  # write it

    '''
    if __name__ == '__main__':
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>>creating images %d/%d' % (i + 1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()
    print('生成完毕')
    '''



#3 image cut out===============
    '''
    将image_preprocess文档的图像分割成四个独立的二值化图像
    ！！！此为第二步
    '''
class image_cut:

    # 加载图像
    def __init__(self):
        self.image_preprocess_get1=image_process()

    def path(self):
        fw = open('.\\data\\img_path.txt', 'r')
        path = fw.readline()
        return path

    def img_solve(self,img_open_path):
        img = self.image_preprocess_get1.img_input(img_open_path)

        gray, blurred = self.image_preprocess_get1.gray_blurred(img)

        binary = self.image_preprocess_get1.getbinary(blurred)

        dilate = self.image_preprocess_get1.getdilate(binary)

        gradient = self.image_preprocess_get1.gradient(dilate)

        thresh1 = self.image_preprocess_get1.getVProjection(dilate)

        self.image_preprocess_get1.image_cut(dilate)

        x, y = self.image_preprocess_get1.confirmpoint(img)

    def show_cut(self):
        cut_path = '.\\cut_image'
        import os
        DIR = cut_path  # 要统计的文件夹
        count_num = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])


    if __name__ == '__main__':
        path = path()
        img_solve(path)






#4 resize image get==============
'''
将切割出来的长方形的图片,修改为正方形，并填充白色背景，
！！！此为第三步
'''
class image_resize:
    def __init__(self):
        pass

    def readimg(self):
        image = cv2.imread('.\\cut_image\\0.jpg')
        return image

    # 按指定图像大小调整尺寸
    def resize_image(self,height=64, width=64):

        # 读取该文件夹内文件的数量
        file_number = len(glob.glob('.\\cut_image\\*.jpg'))

        print('初次调整尺寸的文件数：', file_number)

        for i in range(file_number):
            image = cv2.imread('.\\cut_image\\%d.jpg' % i)  # 读取图像
            top, bottom, left, right = (0, 0, 0, 0)  # 设定四个值
            # 获取图片尺寸：高和宽
            h, w, _ = image.shape
            # 对于长宽不等的图片，找到最长的一边
            longest_edge = max(h, w)
            # 计算短边需要增加多少像素宽度才能与长边等长(相当于padding，长边的padding为0，短边才会有padding)
            if h < longest_edge:
                dh = longest_edge - h  # 最长的边减去高
                top = dh // 2
                bottom = dh - top  # 整个判断部分为确定该填充图像的大小
            elif w < longest_edge:
                dw = longest_edge - w  # 最长的边减去宽
                left = dw // 2
                right = dw - left
            else:
                pass  # pass是空语句，是为了保持程序结构的完整性。pass不做任何事情，一般用做占位语句。

            # RGB颜色
            BLACK = [225, 225, 225]
            # 给图片增加padding，使图片长、宽相等
            # top, bottom, left, right分别是各个边界的宽度，cv2.BORDER_CONSTANT是一种border type，表示用相同的颜色填充
            constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
            # 调整图像大小并返回图像，目的是减少计算量和内存占用，提升训练速度

            cv2.imwrite('.\\cut_image\\%d.jpg' % (i), constant)

        # return cv2.resize(constant, (height, width))

        '''
        if __name__ == '__main__':
            #image=readimg()
            resize_image()
        '''



#5 image packing get=============
'''
再次修改图片尺寸，修改为神经网络接收的图片尺寸28*28
！！！此为第四步
'''
class image_packing():

    def __init__(self):
        pass

    @staticmethod
    def convertjpg(jpgfile, outdir, width=160, height=60):
        # 函数参数类型：图像文件名，图像路径，将图像转换成28*28大小的统一格式
        img = Image.open(jpgfile)  # 第三方库打开图像文件
        try:
            new_img = img.resize((width, height), Image.BILINEAR)  # 重新统一图像尺寸
            new_img = new_img.convert('RGB')
            new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))  # 保存图像
        except Exception as e:
            print(e)

        return new_img

    def image_continue_packing(self):

        file_number = len(glob.glob('.\\screenshot_test\\*.png'))
        print('二次转换尺寸的文件数：', file_number)

        for jpgfile in glob.glob(".\\screenshot_test\\*.png"):
            self.convertjpg(jpgfile, ".\\screenshot_test\\")


    '''
    if __name__ == '__main__':
        image_continue_packing()
    '''






#6 model_judge_image============
class image_judge:
    def __init__(self):
        pass

    number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    CHAR_SET = number

    def img_path_main(self):
        fw = open('.\\data\\img_path.txt', 'r')
        img_path = fw.readline()
        return img_path

    from PIL import Image

    def produceImage(self,file_in):
        width = 160
        height = 60
        image = Image.open(file_in)
        resized_image = image.resize((width, height), Image.ANTIALIAS)
        print(resized_image)
        return resized_image

    model = tf.keras.models.load_model('.\\model440')

    def get_next_batch(self,batch_size=128):
        batch_x = np.zeros([batch_size, 60, 160, 1])
        # batch_y = np.zeros([batch_size, MAX_CAPTCHA, CHAR_SET_LEN])

        img_path = self.img_path_main()

        def wrap_gen_captcha_text_and_image():
            while True:
                # image = Image.open(img_path)
                img_stu = self.produceImage(img_path)
                image = np.array(img_stu)
                if image.shape == (60, 160, 3):
                    return image

        for i in range(batch_size):
            image = wrap_gen_captcha_text_and_image()
            image = tf.reshape(self.convert2gray(image), (60, 160, 1))
            batch_x[i, :] = image

        return batch_x

    def convert2gray(self,img):
        if len(img.shape) > 2:
            gray = np.mean(img, -1)
            return gray
        else:
            return img

    def vec2text(self,vec):
        text = []
        for i, c in enumerate(vec):
            text.append(self.CHAR_SET[c])
        return "".join(text)

    def predict_value(self,):

        data_x = self.get_next_batch(1)
        predict = self.model.predict(data_x)
        predict = self.vec2text(np.argmax(predict, axis=2)[0])

        print(predict)
        return predict

    '''
    if __name__ == '__main__':
    predict_value()
    '''

#7  socket_clien===============

class socket_clien:

    def __init__(self):
        pass

    def getdatanum(self,getdata):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 声明socket类型，同时生成链接对象

        host = socket.gethostname()

        client.connect((host, 27060))  # 建立一个链接，连接到本地的6969端口

        while True:
            # addr = client.accept()
            # print '连接地址：', addr
            msg = getdata  # strip默认取出字符串的头尾空格
            client.send(msg.encode('utf-8'))  # 发送一条信息 python3 只接收btye流
            data = client.recv(10240)  # 接收一个信息，并指定接收的大小 为1024字节
            print('recv:', data.decode())  # 输出我接收的信息
            client.close()  # 关闭这个链接



