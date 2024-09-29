from django.contrib.auth.models import AbstractUser
from django.db import models

# Create your models here.

class User(AbstractUser):
    photo_url = models.CharField('用户头像路径', max_length=128, default='')
    is_login = models.BooleanField('登录状态', default=False)
    is_admin = models.BooleanField('是否为管理员', default=False)

    class Meta:
        db_table = 'Users'

    def __str__(self):
        return self.username

