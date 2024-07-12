from scapy.all import PcapReader
import matplotlib.pyplot as plt

# 替换为实际的pcapng文件路径
pcapng_file = '002.pcapng'

# 存储SSID和信号强度
ssids = []
signal_strengths = []

# 使用Scapy读取pcapng文件
packets = PcapReader(pcapng_file)

# 遍历数据包并提取SSID和信号强度
for pkt in packets:
    if pkt.haslayer("Dot11Beacon"):
        ssid = pkt.info.decode()
        signal_strength = -(256 - pkt.dBm_AntSignal)
        ssids.append(ssid)
        signal_strengths.append(signal_strength)

# 绘制条形图
plt.figure(figsize=(12, 6))
bars = plt.bar(ssids, signal_strengths, color='skyblue', edgecolor='black')  # 设置条形颜色和边缘颜色
plt.xlabel('SSID', fontsize=12)
plt.ylabel('Signal Strength (dBm)', fontsize=12)
plt.title('Signal Strength of Wi-Fi Networks', fontsize=14)

# 添加数值标签和调整字体
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{int(yval)} dBm', ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加水平虚线网格线并设置透明度

# 调整布局和填充
plt.tight_layout(pad=2.0)

# 显示图表
plt.show()
