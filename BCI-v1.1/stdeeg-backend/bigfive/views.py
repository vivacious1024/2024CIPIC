from django.shortcuts import render

# Create your views here.
import subprocess
from django.core.cache import cache
import os
import signal
import sys
import webbrowser
from django.http import JsonResponse

def start_front_process(request):
    try:
        process = subprocess.Popen(["npm.cmd", "run", "dev"], cwd='../big-five/frontend')
        print({'status': 'success'})
        #cache.set('front_process_id', process.pid)        
        webbrowser.open("http://localhost:5174/")
        return JsonResponse({'result': 0, 'message': r'前端打开成功'})
    except Exception as e:
        print({'status': 'error', 'message': str(e)})
        return JsonResponse({'result': 1, 'message': r'前端打卡错误'})

def start_back_process(request):
    try:
        process = subprocess.Popen(["npm.cmd", "run", "dev"], cwd='../big-five/backend')
        #cache.set('back_process_id', process.pid)
        print({'status': 'success'})
        return JsonResponse({'result': 0, 'message': r'后端打开成功'})
    except Exception as e:
        print({'status': 'error', 'message': str(e)})
        return JsonResponse({'result': 1, 'message': r'后端打开错误'})

def start_ai_process(request):
    virtual_env_python_path = '../big-five/backend/python/venv/Scripts/python.exe'
    env = os.environ.copy()
    env['DJANGO_SETTINGS_MODULE'] = 'API.settings' 
    
    try:
        process = subprocess.Popen([virtual_env_python_path, 'manage.py', 'runserver', '0.0.0.0:8080'], cwd='../big-five/backend/python', env=env)
        print({'status': 'success'})
        #cache.set('ai_process_id', process.pid)
        return JsonResponse({'result': 0, 'message': r'算法打开成功'})
    except Exception as e:
        print({'status': 'error', 'message': str(e)})
        return JsonResponse({'result': 1, 'message': r'算法打开错误'})

def kill_port_process(port):
    # 根据操作系统选择不同的命令
    command = ['lsof', '-i', f':{port}'] if sys.platform != "win32" else ['netstat', '-ano', '|', 'findstr', f'{port}']

    try:
        # 执行命令获取占用端口的进程信息
        print('netstat -aon|findstr '+ port)
        with os.popen('netstat -aon|findstr '+ port) as r:
            output = r.read()
        print(output)
        
        # 处理Windows和Unix/Linux系统的输出差异
        if sys.platform == "win32":
            # Windows系统使用tasklist命令获取进程ID
            process_id = output.split('\n')[0].split()[-1]
            print(process_id)
        else:
            # Unix/Linux系统使用lsof命令获取进程ID
            process_id = output.split('\n')[0].split()[1]
        
        # 杀死进程
        if sys.platform == "win32":
            subprocess.call(['taskkill', '-f', '-t', '/pid', process_id])
        else:
            subprocess.call(['kill', '-9', process_id])

        print(f'Process {process_id} using port {port} has been killed.')
    except subprocess.CalledProcessError as e:
        print(f'No process found using port {port}. Error: {e}')
    except Exception as e:
        print(f'An error occurred: {e}')

def stop_process(request):
    # 从缓存获取进程 ID 并终止进程
    '''front_process_id = cache.get('front_process_id')
    print(front_process_id)
    if front_process_id and isinstance(front_process_id, int):
        os.kill(front_process_id, signal.SIGTERM)
        cache.delete('front_process_id')
    
    back_process_id = cache.get('back_process_id')
    print(back_process_id)
    if back_process_id and isinstance(back_process_id, int):
        os.kill(back_process_id, signal.SIGTERM)
        cache.delete('back_process_id')
    
    ai_process_id = cache.get('ai_process_id')
    print(ai_process_id)
    if ai_process_id and isinstance(ai_process_id, int):
        os.kill(ai_process_id, signal.SIGTERM)
        cache.delete('ai_process_id')'''
    
    kill_port_process('5174')
    kill_port_process('8080')
    kill_port_process('3000')
    return JsonResponse({'result': 0, 'message': r'关闭成功'})