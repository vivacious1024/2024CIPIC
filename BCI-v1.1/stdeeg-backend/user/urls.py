from django.urls import path
from user.views import *
from django.contrib import admin

urlpatterns = [
    # path('url_name', api_name)
    # 这是一个样例，指定路由名为url_name，对应处理函数为当前app内views.py中的api_name
    path('register', register),
    path('login', login),
    path('get_usernames', get_all_usernames),
    path('upload_avatar', upload_avatar),
    path('change_user_name', change_user_name),
    path('change_user_email', change_user_email),
    path('get_self_information',get_self_information),

]
