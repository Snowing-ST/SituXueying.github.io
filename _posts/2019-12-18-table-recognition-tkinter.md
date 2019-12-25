---
title: 腾讯云API表格识别+图形界面
author: Situ
layout: post
categories: [work]
tags: [工作实用]
---

原代码参考[腾讯云文字识别API提取表格数据并生成Excel文件 — Young's Blog](http://datascience.org.cn/2019/06/14/%E8%A1%A8%E6%A0%BC%E6%96%87%E5%AD%97%E6%8F%90%E5%8F%96/)，原代码调用腾讯云API进行表格识别，本代码利用Tkinter做了图形界面，可通过pyinstall打包成exe文件，方便不会python的人使用

## 关于tkinter学习资料推荐 

#### 1. 看看大家都怎么学

比如我在知乎上看到[为什么很多Python开发者写GUI不用Tkinter，而要选择PyQt和wxPython或其他？ - Naples的回答 - 知乎](
https://www.zhihu.com/question/32703639/answer/165326590)，这篇回答里提到：

        Tkinter的相关文档：第一份是Tkinter简明教程，不知所云，几乎没什么帮助；第二份是2014年度辛星Tkinter教程第二版，内容浅显易懂；第三份是Python GUI Programming Cookbook，内容详细，帮助很大。

#### 2. 找经典教材

我就在网上找了《Python GUI Programming Cookbook》的[pdf版](https://vdisk.weibo.com/s/qK_5uTIUfrjE?sudaref=www.baidu.com&display=0&retcode=6102)

## 说明
- 先在腾讯云注册账号
- 开通行业文档识别——表格识别
- 在控制台-访问管理-访问密钥，新建密钥，填入代码中
- 表格照片需清晰，不然容易变成乱码，有黑色边框时识别效果最好
- 查询多张时容易遇到ClientNetworkError，原因不明，推荐一张一张查

<img src="{{ 'assets/images/post_images/graph_sample.png'| relative_url }}" /> 
<img src="{{ 'assets/images/post_images/graph_sample2.png'| relative_url }}" /> 

```python
from pandas import Series,DataFrame,ExcelWriter
import os
import re
from json import loads 
from base64 import b64encode

##导入腾讯AI api
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.ocr.v20181119 import ocr_client, models

from tkinter import filedialog,Tk,Label,Button,Menu,Entry

#定义函数,来自于官方文档
def excelFromPictures(path,picture):
    SecretId = ""
    SecretKey = ""     
    
    with open(picture,"rb") as f:
            img_data = f.read()
    img_base64 = b64encode(img_data)
    cred = credential.Credential(SecretId, SecretKey)  #ID和Secret从腾讯云申请
    httpProfile = HttpProfile()
    httpProfile.endpoint = "ocr.tencentcloudapi.com"

    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    client = ocr_client.OcrClient(cred, "ap-shanghai", clientProfile)

    req = models.TableOCRRequest()
    params = '{"ImageBase64":"' + str(img_base64, 'utf-8') + '"}'
    req.from_json_string(params)
#    false=0
    try:

        resp = client.TableOCR(req)
        #     print(resp.to_json_string())

    except TencentCloudSDKException as err:
        print("错误[",err,"]\n可重试")
        


    ##提取识别出的数据，并且生成json
    result1 = loads(resp.to_json_string())

    #RowTl表示数据所有行索引,ColTl表示数据所在列索引,Text为数据
    rowIndex = []
    colIndex = []
    content = []

    for item in result1['TextDetections']:
        rowIndex.append(item['RowTl'])
        colIndex.append(item['ColTl'])
        content.append(item['Text'])

    ##导出Excel
    ##ExcelWriter方案
    rowIndex = Series(rowIndex)
    colIndex = Series(colIndex)

    index = rowIndex.unique()
    index.sort()

    columns = colIndex.unique()
    columns.sort()

    data = DataFrame(index = index, columns = columns)
    for i in range(len(rowIndex)):
        data.loc[rowIndex[i],colIndex[i]] = re.sub(" ","",content[i])

    writer = ExcelWriter(path+"/tables/" +re.match(".*\.",f.name).group()+"xlsx", engine='xlsxwriter')
    data.to_excel(writer,sheet_name = 'Sheet1', index=False,header = False)
    writer.save()
    
    print("已经完成" + f.name + "的提取")

# 查单张 输入表格图片路径
def one_pic():
    #picture_path = input("请输入表格图片路径：")
    picture_path = entry_filename1.get()
    print(picture_path)
    picture_name = os.path.basename(picture_path)
    path = os.path.dirname(picture_path)
    os.chdir(path)
    
    table_path = os.path.join(path,"tables")
    
    if not os.path.exists(table_path):
        os.mkdir(table_path)
        
    excelFromPictures(path,picture_name)
        


# 查多张 一次选择多张图片 本函数将提取图片文件夹的路径
def batch():
    file_str = entry_filename2.get()
#    print(file_str)
    file_names = re.split(r"[{} ]",file_str)
#    print(file_names)
    file_names = [f.lstrip() for f in file_names if f not in [""," "]]
    file_names = [f.rstrip() for f in file_names]
    pictures_path = os.path.dirname(file_names[0])
    path = os.path.dirname(pictures_path)
    os.chdir(pictures_path)
    
    pictures = [os.path.basename(f) for f in file_names]
    
    table_path = os.path.join(path,"tables") 
    
    if not os.path.exists(table_path):
        os.mkdir(table_path)
    
    
    for pic in pictures:
        excelFromPictures(path,pic)
        

window = Tk()
window.title('表格识别神器')  
window.geometry('300x200')

def file_input_one():
    filename = filedialog.askopenfilename(title='导入图片文件')
    entry_filename1.insert('insert', filename) 

def file_input_batch():
    filename = filedialog.askopenfilenames(title='导入图片文件')
    entry_filename2.insert('insert', filename) 

# 菜单：导入文件路径    
menubar = Menu(window) 
filemenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='Open_one_file', command=file_input_one)
filemenu.add_command(label='Open_files', command=file_input_batch)
window.config(menu=menubar)

# 标签
l1 = Label(window, text="单张表格图片识别",font=("宋体", 10, 'bold'))
l1.grid(column=0, row=0)

# 输入框：显示图片文件路径
entry_filename1 = Entry(window, width=30,font=("arial", 10))
entry_filename1.grid(column=0, row=1)

# 按钮：点击运行程序，识别表格图片
b1 = Button(window, text="开始识别",command=one_pic)
b1.grid(column=1, row=1)

l2 = Label(window, text="批量表格图片识别",font=("宋体", 10, 'bold'))
l2.grid(column=0, row=2)

entry_filename2 = Entry(window, width=30,font=("arial", 10))
entry_filename2.grid(column=0, row=3)

b2 = Button(window, text="开始识别",command=batch)
b2.grid(column=1, row=3)

tips = Label(window, text="注：图片名称中不允许有空格",font=("仿宋", 8))
tips.grid(column=0,row=4)

window.mainloop()
```
