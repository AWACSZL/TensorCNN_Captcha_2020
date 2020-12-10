# -*- coding: UTF-8 -*-
# coding: utf-8
import tkinter as tk
import threading
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from GUI_Utils import LayoutGUI
from PIL import Image, ImageTk
import os
import sys
import difflib
import cv2
import time
from tkinter import messagebox
import random
import App_Core


#GUI界面
class captchagui:
    #=====全局设定============
    job: threading.Thread
    is_task_running: bool = False


    def __init__(self,root:tk.Tk):
        # layout帮助进行位置偏移
        self.layout = {
            'global': {
                'start': {'x': 15, 'y': 20},
                'space': {'x': 15, 'y': 25},
                'tiny_space': {'x': 5, 'y': 10}
            }
        }
        self.root=root
        self.root.title('基于深度学习的验证码识别平台')
        self.window_width=500
        self.window_height=600
        screenwidth,screenheight = self.root.maxsize()
        self.root.geometryparam='%dx%d+%d+%d'%(self.window_width, self.window_height, (screenwidth-self.window_width)/2, (screenheight - self.window_height)/2)
        self.root.geometry(self.root.geometryparam)
        #self.root.wm_attributes('-topmost',1)  #窗口置顶
        self.layout_utils = LayoutGUI(self.layout, self.window_width)

        self.sysout=sys.stdout
        #self.text = Text(self.root)
        #self.text.pack()
        self.root.resizable(0,0)  #固定大小


#=========================菜单栏============================#

        #菜单栏全局设置
        self.menubar = tk.Menu(self.root)
        self.filemenu = tk.Menu(self.menubar, tearoff=False)
        self.testmenu = tk.Menu(self.menubar, tearoff=False)
        self.help_menu = tk.Menu(self.menubar, tearoff=False)
        self.editmenu = tk.Menu(self.menubar, tearoff=False)
        #self.edit_var = tk.DoubleVar()

        self.memory_usage_menu = tk.Menu(self.menubar, tearoff=False)

        #菜单栏添加子菜单大类文件/filemenu
        self.menubar.add_cascade(label="文件", menu=self.filemenu)
        #子菜单大类文件/filemenu添加各小类
        self.filemenu.add_cascade(label="打开文件",command=lambda:self.opeimg())
        #添加分割线
        self.filemenu.add_separator()
        self.filemenu.add_cascade(label='退出',command=lambda:self.root.destroy())


        #菜单栏添加子菜单测试/test
        self.menubar.add_cascade(label="编辑", menu=self.testmenu)
        self.testmenu.add_command(label='开始识别',command=lambda:self.start_test())
        self.testmenu.add_separator()
        self.testmenu.add_command(label='生成验证码', command=lambda: self.creat_captcha())
        self.testmenu.add_separator()
        self.testmenu.add_command(label='输出日志',command=lambda:self.show_log())
        self.testmenu.add_separator()
        self.testmenu.add_command(label='随机验证码', command=lambda: self.random_captcha())
        self.testmenu.add_separator()
        self.testmenu.add_command(label='裁剪图像', command=lambda: self.image_cut())

        # 菜单栏添加子菜单大类帮助/helpmenu
        self.menubar.add_cascade(label="帮助", menu=self.help_menu)
        # 子菜单大类编辑/helpmenu添加各小类
        self.help_menu.add_command(label="帮助", command=lambda:self.help_gui())
        self.help_menu.add_command(label="关于", command=lambda: self.about_gui())

        # 菜单栏添加子菜单大类编辑/editmenu
        self.menubar.add_cascade(label="接口", menu=self.editmenu)
        # 子菜单大类编辑/editmenu添加各小类
        self.editmenu.add_command(label="测试1", command=lambda: self.test())
        self.editmenu.add_command(label="测试2", command=lambda: self.test())
        # 添加分割线
        self.editmenu.add_separator()
        self.editmenu.add_command(label="测试3", command=lambda: self.socket_send())


        self.root.config(menu=self.menubar)


#===================group1原始验证码打开显示框==========================

        #组1外部容器框架
        self.label_frame_source=ttk.LabelFrame(self.root,text='captcha show')
        self.label_frame_source.place(
            x=self.layout['global']['start']['x'],
            y=self.layout['global']['start']['y'],
            width=455,
            height=90,
        )

        ## 容器内标签实例
        #self.dataset_train_path_text = ttk.Label(self.root, text='原始图像',font=('微软雅黑', 10), anchor=tk.W)
        #self.layout_utils.inside_widget(
        #    src=self.dataset_train_path_text,
        #    target=self.label_frame_source,
        #    width=150,
        #    height=70
        #)


#===================group2图像处理过程显示框============================
        self.label_frame_neu = ttk.Labelframe(self.root, text='process control show')
        self.layout_utils.below_widget(
            src=self.label_frame_neu,
            target=self.label_frame_source,
            width=455,
            height=240,
            tiny_space=False
        )




#===================group3各类数据显示框=============================
        self.label_frame_train = ttk.Labelframe(self.root, text='data processing time')

        self.layout_utils.below_widget(
            src=self.label_frame_train,
            target=self.label_frame_neu,
            width=455,
            height=70,
            tiny_space=True
        )

        #控制台终端数据显示

       # self.dataset_train_listbox = tk.Listbox(self.root, font=('微软雅黑', 9))
       # self.layout_utils.next_to_widget(
       #     src=self.dataset_train_listbox,
       #     target=self.label_frame_train,   #######
       #     width=640,
       #     height=36,
       #     tiny_space=False
       # )

#===========================group4结果显示=========================
        self.label_frame_data_set = ttk.Labelframe(self.root, text='Results display')

        self.layout_utils.below_widget(
            src=self.label_frame_data_set,
            target=self.label_frame_train,
            width=455,
            height=75,
            tiny_space=False
        )

        #这个会根据容器变化而变化，但是位置不对
        #label_edge = self.layout_utils.object_edge_info(self.dataset_train_path_text)
        #widget_edge = self.layout_utils.object_edge_info(self.dataset_train_listbox)
        #self.dataset_validation_path_text = ttk.Label(self.root, text='验证结果', anchor=tk.W)
        #self.dataset_validation_path_text.place(
        #    x=label_edge['x'],
        #    y=widget_edge['edge_y'] + self.layout['global']['space']['y'] / 2,
        #    width=100,
        #    height=20
        #)

#===========================主要组件==============================
#标签大类

    #原始图片部分
        #原始图像
        self.label = Label(self.root, text='原始图像', font=('微软雅黑', 10)).place(relx=0.06, rely=0.10)
        #展示数字
        self.label = Label(self.root, text='所示字符', font=('微软雅黑', 10)).place(relx=0.60, rely=0.10)


    #结果显示部分
        #结果标签
        self.label = Label(self.root, text='验证结果',font=('微软雅黑', 10)).place(relx=0.06, rely=0.85)
        #准确率标签
        self.label = Label(self.root, text='准确率', font=('微软雅黑', 10)).place(relx=0.52, rely=0.85)

#按钮大类

    #开始识别
        self.button = tk.Button(self.root,text='开始识别',font=('微软雅黑',8),command=lambda:self.start_test(),width=10,heigh=1).place(relx=0.38,rely=0.94)
    #随机验证码
        self.button = Button(self.root, text='随机验证码', font=('微软雅黑', 8), command=lambda:self.random_captcha(), width=10, heigh=1).place(relx=0.58, rely=0.94)
    #退出
        self.button = Button(self.root, text='清空退出', font=('微软雅黑', 8), command=lambda: self.root.destroy(), width=10,heigh=1).place(relx=0.78, rely=0.94)
    ##随机选择验证码
    #    self.button = Button(self.root, text='随机验证码', font=('微软雅黑', 8), command=lambda:self.random_captcha(), width=10, heigh=1).place(relx=0.78, rely=0.94)
    ##清空退出
    #    self.button = Button(self.root, text='清空退出', font=('微软雅黑', 8), command=lambda:self.root.destroy(), width=10, heigh=1).place(relx=0.80, rely=0.94)



        self.image_preprocess_get1 = App_Core.image_process()
        self.model_judge_image = App_Core.image_judge()
        self.image_packing = App_Core.image_packing()
        self.socket_clien = App_Core.socket_clien()
        self.creat_Captcha_get = App_Core.create_captcha()
        self.image_cutting = App_Core.image_cut()
        self.image_resize = App_Core.image_resize()
        #self.convertjpg=App_core.image_packing.convertjpg()

#===========================各类触发函数=========================


    def test(self):
        messagebox.showinfo('测试','该功能仅作程序测试使用')
        print('触发空值测试函数')


    def opeimgpath(self):
        print('打开图像路径记录函数')
        global realimgpath
        realimgpath = tk.filedialog.askopenfilename(filetypes=[("PNG", ".png"),("JPG", ".jpg"), ("GPF", ".gpf")])
        print('正在执行写入到图像路径文本')
        fw=open('.\\data\\img_path.txt','w')
        fw.write(realimgpath)

        print('图像路径：',realimgpath)
        return realimgpath


    def opeimg(self):
        print('打开图像路径文本')
        imgpath=self.opeimgpath()
        print('IMAGE函数调用图像')
        global img_stu, label_img
        img_open = Image.open(imgpath)
        img_stu = self.resizeimg(160, 60,img_open)
        img_stu = ImageTk.PhotoImage(img_stu)

        #所显示的原始字符
        file_name_add = os.path.basename(imgpath)
        print(file_name_add)
        file_name = file_name_add.split('.')[0]
        print(file_name)
        label = Label(self.root, text=file_name+'\t\t', font=('微软雅黑', 20))
        label.place(relx=0.73, rely=0.09)

        #所显示的原始验证码图片
        label_img = Label(self.root, image=img_stu)
        label_img.place(relx=0.20, rely=0.07)
        print(img_open)

        return img_stu


    def resizeimg(self,imgw,imgh,img_open):
        print('调用修改图像尺寸函数')
        w, h = img_open.size
        f1 = 1.0 * imgw / w
        f2 = 1.0 * imgh / h
        factor = min([f1, f2])
        width = int(w * factor)
        height = int(h * factor)
        return img_open.resize((width, height), Image.ANTIALIAS)


    #def show_data_all(self):
    #    while True:
    #        self.text.insert(END, '...')
    #        self.root.update()  # 更新以后才能看到变化
    #        time.sleep(1)  # 这里为了快点看到效果，改为了1S输出一次

    ##展示原始图片的字符/目前没有函数调用
    #def show_file_name(self):
    #    file_name_add = os.path.basename(realimgpath)
    #    print(file_name_add)
    #    file_name = file_name_add.split('.')[0]
    #    print(file_name)
    #    label=Label(self.root,text=file_name,font=('微软雅黑',8))
    #    label.place(relx=0.55,rely=0.3)
    #    return file_name


    def process_img(self):

        print('正在调用图像处理函数')

        fw=open('.\\data\\img_path.txt','r')
        path=fw.readline()
        count_num=len(path)

        print('图像路径长度为：',count_num)

        if count_num==0:

            messagebox.showinfo('提示','需要先打开原图片')
            print('打开图像失败')
        else:
            source_img=self.image_preprocess_get1.img_input(path)

            gray_img,blurred_img=self.image_preprocess_get1.gray_blurred(source_img)
            print('正在将图像灰度化，模糊化...')
            cv2.imwrite('.\\process_image\\' + 'gray.jpg', gray_img)
            print('cv将灰度化图像写入...')
            cv2.imwrite('.\\process_image\\' + 'blurred.jpg', blurred_img)
            print('cv将模糊化图像写入')
            binary_img=self.image_preprocess_get1.getbinary(blurred_img)
            print('正在将图像二值化...')
            cv2.imwrite('.\\process_image\\' + 'binary.jpg', binary_img)
            print('cv将二值化图像写入...')
            dilate_img=self.image_preprocess_get1.getdilate(binary_img)
            print('正在将图像膨胀...')
            cv2.imwrite('.\\process_image\\' + 'dilate.jpg', dilate_img)
            print('cv将膨胀化图像写入...')
            gradient_img=self.image_preprocess_get1.gradient(dilate_img)
            print('正在将图像梯度化...')
            cv2.imwrite('.\\process_image\\' + 'gradient.jpg', gradient_img)
            print('cv将梯度化图像写入...')
            thresh_img=self.image_preprocess_get1.getVProjection(binary_img)
            print('正在建图像垂直分割...')
            cv2.imwrite('.\\process_image\\' + 'thresh.jpg', thresh_img)
            print('cv将垂直分割图像写入...')

            #image_preprocess_get1.image_cut(binary_img)
            #image_cutout_get2.img_solve(path)
            #resize_img_get3.resize_image()
            #image_packing_get4.image_continue_packing()


            #神经网络识别数字
            print('开始执行神经网络模型识别...')
            cnn_start_time=time.time()

            get_num=self.model_judge_image.predict_value()
            str_get_num=[str(i) for i in get_num]
            str_get_num=''.join(str_get_num)
            print(str_get_num)
            fw=open('.\\data\\result.txt','w')
            fw.write(str_get_num)
            result_label=Label(self.root,text=str_get_num,font=('微软雅黑', 20))
            result_label.place(relx=0.22, rely=0.84)

            cnn_end_time=time.time()
            print('神经网络模型识别结束...')
            return cnn_start_time,cnn_end_time


    def loss_value(self):

        print('#========以下为模型判断与准确率函数的输出==========#')

        fw = open('.\\data\\img_path.txt', 'r')
        path = fw.readline()
        print(path)
        file_name_add = os.path.basename(path)
        print(file_name_add)
        source_num = file_name_add.split('.')[0]

        fw=open('.\\data\\result.txt','r')
        result=fw.readline()

        loss=self.str_compare(source_num,result)



        print(source_num)
        print(result)
        print(loss)

        loss_label=Label(self.root,text='%01f'%loss,font=('微软雅黑', 20))
        loss_label.place(relx=0.64, rely=0.84)

    def str_compare(self,str1,str2):

        return difflib.SequenceMatcher(None, str1, str2).quick_ratio()





    def start_test(self):

        self.image_packing.image_continue_packing()

        start_time=time.time()

        start_core_time,end_core_time=self.process_img()

        self.show_procress()

        self.loss_value()

        #self.clean_data()

        end_time=time.time()

        self.work_time(start_time,end_time,start_core_time,end_core_time)


    def show_procress(self):

        global gray,gray_img,gray_label_img
        #灰度化图像展示
        gray=Image.open('.\\process_image\\gray.jpg')
        gray_img = self.resizeimg(160, 60, gray)
        gray_img = ImageTk.PhotoImage(gray_img)
        gray_label_img = Label(self.root, image=gray_img)
        gray_label_img.place(relx=0.10, rely=0.28)
        print(gray)
        print(gray_img)
        print(gray_label_img)

        global blurred,blurred_img,blurred_label_img
        #模糊化图像展示
        blurred=Image.open('.\\process_image\\blurred.jpg')
        blurred_img = self.resizeimg(160, 60, blurred)
        blurred_img = ImageTk.PhotoImage(blurred_img)
        blurred_label_img = Label(self.root, image=blurred_img)
        blurred_label_img.place(relx=0.50, rely=0.28)
        print(blurred)
        print(blurred_img)
        print(blurred_label_img)

        global binary,binary_img,binary_label_img
        #二值化图像展示
        binary=Image.open('.\\process_image\\binary.jpg')
        binary_img = self.resizeimg(160, 60, binary)
        binary_img = ImageTk.PhotoImage(binary_img)
        binary_label_img = Label(self.root, image=binary_img)
        binary_label_img.place(relx=0.10, rely=0.40)
        print(binary)
        print(binary_img)
        print(binary_label_img)

        #腐蚀化图像展示
        global dilate,dilate_img,dilate_label_img

        dilate=Image.open('.\\process_image\\dilate.jpg')
        dilate_img = self.resizeimg(160, 60, dilate)
        dilate_img = ImageTk.PhotoImage(dilate_img)
        dilate_label_img = Label(self.root, image=dilate_img)
        dilate_label_img.place(relx=0.50, rely=0.40)
        print(dilate)
        print(dilate_img)
        print(dilate_label_img)


        #梯度化图像展示
        global gradient,gradient_img,gradient_label_img

        gradient=Image.open('.\\process_image\\gradient.jpg')
        gradient_img = self.resizeimg(160, 60, gradient)
        gradient_img = ImageTk.PhotoImage(gradient_img)
        gradient_label_img = Label(self.root, image=gradient_img)
        gradient_label_img.place(relx=0.10, rely=0.52)
        print(gradient)
        print(gradient_img)
        print(gradient_label_img)

        #像素区块展示
        global thresh,thresh_img,thresh_label_img
        thresh=Image.open('.\\process_image\\thresh.jpg')
        thresh_img = self.resizeimg(160, 60, thresh)
        thresh_img = ImageTk.PhotoImage(thresh_img)
        thresh_label_img = Label(self.root, image=thresh_img)
        thresh_label_img.place(relx=0.50, rely=0.52)
        print(thresh)
        print(thresh_img)
        print(thresh_label_img)


    def work_time(self,stime,etime,sctime,ectime):
        global work_time,core_work_time,time_label,core_time_label
        work_time=etime-stime
        time_label=Label(self.root,text='程序运行总时间和：%f 秒(seconds)'%work_time,font=('微软雅黑',10))
        time_label.place(relx=0.06,rely=0.67)

        core_work_time=ectime-sctime
        core_time_label=Label(self.root,text='核心模型识别时间：%f 秒(seconds)'%core_work_time,font=('微软雅黑',10))
        core_time_label.place(relx=0.06, rely=0.71)

        print('#====以下为程序运行时间输出测试=====#')
        print('总程序开始时间：',stime)
        print('总程序完成时间',etime)
        print('模型开始时间',sctime)
        print('模型完成时间',ectime)
        print('总程序运行时间',work_time)
        print('模型运行时间',core_work_time)

    def help_gui(self):

        fw = open('.\\data\\help.txt','r',encoding='utf-8')

        txt = fw.read()
        messagebox.showinfo('帮助',txt)
        print('用户打开了帮助文档')

    def about_gui(self):

        fw = open('.\\data\\about.txt', 'r',encoding='utf-8')
        txt = fw.read()
        messagebox.showinfo('关于',txt)
        print('用户打开了关于文档')

    def clean_data(self):
        process_path='.\\process_image'
        cut_image_path='.\\cut_image'

        #清空处理图片的文件夹
        for root, dirs, files in os.walk(process_path):
            for name in files:
                if name.endswith(".jpg"):  # 指定要删除的格式，这里是jpg 可以换成其他格式
                    os.remove(os.path.join(root, name))
                    print("Delete File: " + os.path.join(root, name))

        # 清空处理图片的文件夹
        for root, dirs, files in os.walk(cut_image_path):
            for name in files:
                if name.endswith(".jpg"):  # 指定要删除的格式，这里是jpg 可以换成其他格式
                    os.remove(os.path.join(root, name))
                    print("Delete File: " + os.path.join(root, name))

        #清空几个txt文件的内容
        #open('img_path.txt','w').close()
        #print('清空原图像路径')
        #open('location.txt','w').close()
        #print('清空分割点位置信息')
        #open('log.txt','w').close()
        #print('清空日志信息')
        #open('result.txt','w').close()
        #print('清空识别结果信息')

    def show_log(self):

        fw=open('.\\data\\log.txt','r')
        log_txt=fw.read()
        messagebox.showinfo('日志', log_txt)
        print('日志显示===')

    def random_captcha(self):

        print('#=========以下为随机选取验证码功能输出测试=========#')
        print('开始执行随机选取验证码函数')
        global img_path,randomfile,random_path,random_image,img_stu,label_img,label,file_name,file_name_add

        img_path='.\\image\\'

        print('导入随机选取验证码图片路径')

        for dir_info in os.walk(img_path):
            dir_name, _, file_names = dir_info
            temp = []
            for file_name in file_names:
                temp.append(file_name)
            if len(temp) == 0:
                continue

            randomfile = random.choice(temp)

            random_path=os.path.join(dir_name, randomfile)

            random_image = Image.open(random_path)

            print('随机选取验证码路径',random_path)
            print('随机选取验证码图片参数',random_image)

            fw = open('.\\data\\img_path.txt', 'w')
            fw.write(random_path)

            img_stu = self.resizeimg(160, 60, random_image)
            img_stu = ImageTk.PhotoImage(img_stu)
            label_img = Label(self.root, image=img_stu)
            label_img.place(relx=0.20, rely=0.07)

            # I.save(os.path.join(dstPath, temp[0]))
            file_name_add = os.path.basename(random_path)
            print(file_name_add)
            file_name = file_name_add.split('.')[0]
            print(file_name)
            label = Label(self.root, text=file_name, font=('微软雅黑', 20))
            label.place(relx=0.73, rely=0.09)


    def socket_send(self):
        print('开始执行socket通讯')
        fw=open('.\\data\\result.txt','r')
        data=fw.readline()
        self.socket_clien.getdatanum(data)

    def creat_captcha(self):
        num = 500
        for i in range(num):
            self.creat_Captcha_get.gen_captcha_text_and_image()
            sys.stdout.write('\r>>creating images %d/%d' % (i + 1, num))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        print('生成完毕')

    def image_cut(self):
        self.clean_data()
        path=self.image_cutting.path()
        self.image_cutting.img_solve(path)
        self.image_resize.resize_image()



if __name__ == '__main__':
    root=tk.Tk()
    maingui=captchagui(root)
    root.mainloop()