from django.urls import path
from bigfive.views import *

urlpatterns = [
    # path('url_name', api_name)
    # 这是一个样例，指定路由名为url_name，对应处理函数为当前app内views.py中的api_name
    path('start_front_process', start_front_process),
    path('start_back_process', start_back_process),
    path('start_ai_process', start_ai_process),
    path('stop_process', stop_process),

]
