# 便携式设备两导脑机接口平台

## 项目简要介绍

stdeeg-frontend是项目前端，使用electron框架，其中渲染进程使用vue框架。其中driver.js，resolve.js负责处理脑电设备的连接以及脑电信号的处理（包含字节流的校验及字节流向脑电信号的转化）。
目前采集模式问题较大，完成度较低，后续可以参考博瑞康的系统完善。  
（另：前端项目启动时主进程会尝试连接一个服务器，服务器位于server，目前完成度非常低。连接成功后如果主进程接收到渲染进程的消息会发送给服务器，服务器会转发给其他客户端。之后如果有其他算法模块需要实时脑电数据的话可能会使用到）

stdeeg-backend是项目后端，使用django，负责管理用户信息以及启动大五人格系统。

启动大五人格功能后，前端调用api向后端端口发送消息，后端打开大五人格系统的前端后端及算法模块

大五人格前端使用vue，后端使用js，算法模块使用python。前端负责字节流转发到后端、展示范式、展示结果，后端负责字节流的校验、字节流向脑电信号的转化、滤波以及存储等，算法模块负责预测人格

## 环境配置与项目启动

### 前端stdeeg-frontend

``npm install`` 安装依赖

``npm run dev``开发模式启动项目

### 后端stdeeg-backend
SA_backend/settings.py中DATABASES请按照本地数据库设置更改


``python -m venv venv``创建虚拟环境

``.\venv\Scripts\activate``激活

``pip install -r requirements.txt``安装依赖

``python manage.py runserver 0.0.0.0:8000``启动项目

### 大五人格前端big-five/frontend
``npm install`` 安装依赖

### 大五人格后端big-five/backend
``npm install`` 安装依赖

### 大五人格算法模块big-five/backend/python

``python -m venv venv``创建虚拟环境（3.8版本）

``.\venv\Scripts\activate``激活

``pip install -r requirements.txt``安装依赖

## 使用流程

打开系统：在stdeeg-frontend目录下运行``npm run dev``,在stdeeg-backend目录下运行``python manage.py runserver 0.0.0.0:8000``

选择被试->添加被试->返回首页->选择设备和端口进入大五人格->弹出网页后先关掉弹出来的翻译请求，没有就忽略->连接设备
->待下方显示连接成功（设备灯由黄变绿）后开始采集->观察范式

## 注意事项

1、数据会保存在data目录中对应的用户名目录下，如果没有选择被试会保存在data/temp目录下

2、打开系统后，第一次点 关于我们 会有一点卡顿

3、关于我们 里面有链接还没有实现

4、如果遇到bug后重新启动项目即可