from django.conf import settings


import os
import random

from django.contrib.auth.hashers import check_password, make_password
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.authentication import TokenAuthentication
from rest_framework.authtoken.models import Token
from rest_framework.decorators import authentication_classes, permission_classes, api_view
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone

from SA_backend.settings import BASE_DIR
from user.models import User


# Create your views here.
def register(request):
    username = request.POST.get('username', '')
    email = request.POST.get('email', '')
    if not username:
        return JsonResponse({'result': 1, 'message': r'缺少必要信息'})

    if User.objects.filter(username=username).exists():
        result = {'result': 1, 'message': r'用户名已存在'}
        return JsonResponse(result)

    user = User.objects.create(username=username, email=email)
    user.save()
    result = {'result': 0, 'message': r'注册成功'}
    return JsonResponse(result)

def login(request):
    username = request.POST.get('username')
    if not User.objects.filter(username=username):
        result = {'result': 1, 'message': r'用户名错误'}
        return JsonResponse(result)

    try:
        user = User.objects.get(username=username)
        token, created = Token.objects.get_or_create(user=user)
        return JsonResponse({'result': 0, 'message': r'登录成功', "token": str(token.key), 'is_admin': user.is_admin})
    except User.DoesNotExist:
        return JsonResponse({'result': 1, 'message': r'未知错误'})

@api_view(['GET'])
def get_all_usernames(request):
    users = User.objects.all().values('username')
    usernames = [user['username'] for user in users]
    result = {'result': 0, 'usernames': usernames}
    return JsonResponse(result)


from django.core.files.storage import FileSystemStorage

@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def upload_avatar(request):
    user = request.user
    avatar = request.FILES.get('photo')

    if not avatar:
        return JsonResponse({'result': 1, 'message': '请上传头像文件'}, status=400)

    # 文件类型验证
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    _, ext = os.path.splitext(avatar.name)
    if ext.lower() not in valid_extensions:
        return JsonResponse({'result': 1, 'message': '无效的文件类型，支持的类型: jpg, jpeg, png, gif'}, status=400)

    # 文件大小限制
    if avatar.size > 5 * 1024 * 1024:
        return JsonResponse({'result': 1, 'message': '文件大小不能超过5MB'}, status=400)

    # 使用 Django 的文件存储
    fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'avatars'))
    filename = fs.save(f'{user.id}_avatar{ext}', avatar)
    
    # 手动构建 URL
    avatar_url = f'/media/avatars/{filename}'

    # 更新用户的头像路径
    user.photo = avatar_url  # 假设你的 User 模型有一个 'photo' 字段
    user.save()

    return JsonResponse({'result': 0, 'message': '上传成功', 'photo_url': avatar_url})

@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def change_user_name(request):
    if request.method == 'POST':
        user = request.user
        new_name = request.data.get('username')

        try:
            user.username = new_name
            user.save()
            return JsonResponse({'result': 0, 'message': 'User name updated successfully.'})
        except User.DoesNotExist:
            return JsonResponse({'error': 'User not found.'}, status=404)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def change_user_email(request):
    if request.method == 'POST':
        user = request.user
        new_email = request.data.get('email')

        try:
            # print(user.email)
            # print(new_email)
            user.email = new_email
            user.save()
            return JsonResponse({'result': 0, 'message': 'User email updated successfully.'})
        except User.DoesNotExist:
            return JsonResponse({'error': 'User not found.'}, status=404)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)



@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def get_self_information(request):
    if request.method == 'GET':
        user = request.user
        try:
            response_data = {
                'id': user.id,
                'username': user.username,
                'last_login': user.last_login,
                'is_superuser': user.is_superuser,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'email': user.email,
                'is_staff': user.is_staff,
                'is_active': user.is_active,
                'date_joined': user.date_joined,
                'photo_url': user.photo.url if user.photo else None,
                'is_login': user.is_login,
                'is_admin': user.is_admin,
                'result': 0,
            }
            return JsonResponse(response_data)
        except User.DoesNotExist:
            return JsonResponse({'error': 'User not found.'}, status=404)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def authentication(request):
    if request.method == 'POST':
        user = request.user
        true_name = request.POST.get('true_name')
        ID = request.POST.get('ID')
        institution = request.POST.get('institution')
        request.user.true_name = true_name
        # request.user.ID = ID
        request.user.institution = institution
        request.user.is_authentication = True
        request.user.save()
        return JsonResponse({'result': 0, 'message': '认证成功'})


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def get_specific_information(request):
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        user = User.objects.get(id=user_id)

        try:
            response_data = {
                'id': user.id,
                'username': user.username,
                'last_login': user.last_login,
                'is_superuser': user.is_superuser,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'email': user.email,
                'is_staff': user.is_staff,
                'is_active': user.is_active,
                'date_joined': user.date_joined,
                'photo_url': user.photo.url if user.photo else None,
                'is_login': user.is_login,
                'is_admin': user.is_admin,
                'is_author': user.is_author,
                'result': 0,
                'photo_url_out': user.photo_url_out,
                'author_id': "",
                'is_authentication': user.is_authentication,
                'true_name': user.true_name,
            }
            author_user = Author_User.objects.filter(user=user).first()
            if author_user:
                response_data['author_id'] = author_user.author_id
            return JsonResponse(response_data)
        except User.DoesNotExist:
            return JsonResponse({'error': 'User not found.'}, status=404)

    return JsonResponse({'error': 'Invalid request method.'}, status=400)
