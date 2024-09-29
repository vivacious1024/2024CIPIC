import socket
import threading
import sys
# 服务器的IP地址和端口
HOST = '0.0.0.0'
PORT = 8712

# 存储所有连接的客户端及其地址
clients = []

def broadcast(message, sender_socket):
    """将消息广播给除消息来源以外的所有连接的客户端"""
    for client_socket, _ in clients:
        if client_socket != sender_socket:
            try:
                client_socket.send(message)
                print(f"向 {client_socket.getpeername()} 发送消息: {message}")
            except:
                print(f"与客户端通信出错，断开连接: {client_socket.getpeername()}")
                client_socket.close()
                clients.remove((client_socket, _))

def handle_client(client_socket):
    """处理客户端连接的函数"""
    while True:
        try:
            # 接收数据
            message = client_socket.recv(1024)
            if not message:
                break
            # 广播给其他客户端
            broadcast(message, client_socket)
        except:
            print(f"与客户端 {client_socket.getpeername()} 通信出错")
            client_socket.close()
            clients.remove((client_socket, client_socket.getpeername()))
            break
    client_socket.close()

def start_server():
    """启动服务器"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()

        print(f"服务器正在监听 {HOST}:{PORT}")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"客户端 {addr} 已连接")

            # 将客户端及其地址添加到列表
            clients.append((client_socket, addr))

            # 创建一个线程来处理客户端
            client_thread = threading.Thread(target=handle_client, args=(client_socket,))
            client_thread.start()

if __name__ == "__main__":
    start_server()