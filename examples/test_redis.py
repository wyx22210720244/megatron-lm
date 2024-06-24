import redislite
import socket

# 获取主机名
hostname = socket.gethostname()
print(f"Hostname: {hostname}")

# 解析主机名为IP地址
ip_address = socket.gethostbyname(hostname)
print(f"IP Address: {ip_address}")

# 解析特定的主机名为IP地址
pod_name = "wyx-master-0"
try:
    pod_ip = socket.gethostbyname(pod_name)
    print(f"IP Address for {pod_name}: {pod_ip}")
except socket.error as err:
    print(f"Error resolving {pod_name}: {err}")

# 使用默认配置启动 redislite
try:
    store = redislite.Redis()
    print("Redis server started successfully")

    # 设置一个键值对
    store.set('name', 'ChatGPT')

    # 获取键的值
    value = store.get('name')
    print(f'The value of "name" is: {value.decode("utf-8")}')
except Exception as e:
    print(f"Failed to start Redis server: {e}")
