import socket
import time

def send_data(client_socket):
    for i in range(500):
        data = f"Data packet {i+1}"
        client_socket.send(data.encode())
        time.sleep(0.002)  # Adjust the sleep time if needed

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 1234))
    server_socket.listen(1)
    print("Server started. Listening for connections...")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Client connected: {client_address}")

        send_data(client_socket)

        client_socket.close()
        print("Data sent. Client disconnected.")

if __name__ == '__main__':
    main()