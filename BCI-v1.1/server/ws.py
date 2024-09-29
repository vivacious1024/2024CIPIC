import asyncio
import websockets
import socket

TCP_HOST = 'localhost'
TCP_PORT = 8712

async def handle_websocket(websocket, path):
    # 创建到TCP服务器的连接
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((TCP_HOST, TCP_PORT))
    
    # 创建一个从TCP到WebSocket的转发任务
    async def tcp_to_ws():
        while True:
            data = tcp_socket.recv(1024)
            if not data:
                break
            await websocket.send(data)
    
    # 启动TCP到WebSocket的转发
    forward_task = asyncio.create_task(tcp_to_ws())
    
    try:
        # 从WebSocket接收消息并转发到TCP
        async for message in websocket:
            tcp_socket.send(message)
    finally:
        # 关闭TCP连接和转发任务
        tcp_socket.close()
        forward_task.cancel()
        await forward_task

# 启动WebSocket服务器
start_server = websockets.serve(handle_websocket, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()