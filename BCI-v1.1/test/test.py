from flask import Flask
import subprocess

# 指定虚拟环境中的python解释器路径
virtual_env_python_path = '../big-five/backend/python/venv/Scripts/python.exe'

try:
    subprocess.Popen([virtual_env_python_path, 'manage.py', 'runserver', '0.0.0.0:8080'], cwd='../big-five/backend/python')
    print({'status': 'success'})
except Exception as e:
    print({'status': 'error', 'message': str(e)})

try:
    subprocess.Popen(["npm.cmd", "run", "dev"], cwd='../big-five/backend')
    print({'status': 'success'})
except Exception as e:
    print({'status': 'error', 'message': str(e)})