"""
URL configuration for SA_backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.contrib import admin
from django.urls import path

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # path('', index),
    # path('search/', PublisherDocumentView.as_view({'get': 'list'})),
    path('api/user/', include(('user.urls', 'user'))),
    path('api/bigfive/', include(('bigfive.urls', 'bigfive'))),
    #path('api/Administrator/', include(('Administrator.urls', 'Administrator'))),
    #path('api/browhistory/',include(('Browhistory.urls', 'Browhistory'))),
    #path('api/academia/', include(('Academia.urls', 'Academia'))),
    #path('api/message/', include(('message.urls', 'message'))),
    # path('api/Academia/', include(('Academia.urls', 'Academia'))),

    path('admin/', admin.site.urls),
]
# 添加媒体文件的 URL 路由
if settings.DEBUG:  # 仅在开发模式下服务媒体文件
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)