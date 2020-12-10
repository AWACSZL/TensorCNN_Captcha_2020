#!/usr/bin/python

import tkinter as tk
import socket


rootwd=tk.Tk()
rootwd.title('协同平台接收端')
rootwd.resizable(True, True)
windowWidth = 400
windowHeight = 250

screenWidth,screenHeight = rootwd.maxsize()     #获得屏幕宽和高
geometryParam = '%dx%d+%d+%d'%(windowWidth, windowHeight, (screenWidth-windowWidth)/2, (screenHeight - windowHeight)/2)
rootwd.geometry(geometryParam)       #设置窗口大小及偏移坐标



# 建立一个服务端
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host = socket.gethostname()
server.bind((host,27060)) #绑定要监听的端口
server.listen(5) #开始监听 表示可以使用五个链接排队
while True:# conn就是客户端链接过来而在服务端为期生成的一个链接实例
    conn,addr = server.accept() #等待链接,多个链接的时候就会出现问题,其实返回了两个值
    print(conn,addr)
    while True:
        try:
            data = conn.recv(10240)  #接收数据
            print('recive:',data) #打印接收到的数据
            labelx=tk.Label(rootwd,text='接收为:'+data.decode('utf-8'),width=15,heigh=2,font=('宋体',30),background='red').place(x=20,y=20)
            rootwd.mainloop()
            conn.send(data.upper()) #然后再发送数据
            break
        except ConnectionResetError as e:
            print('关闭了正在占线的链接！')
            break
        conn.close()


rootwd.mainloop()
