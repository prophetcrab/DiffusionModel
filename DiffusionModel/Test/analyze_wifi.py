import pyshark
import pandas as pd
import matplotlib.pyplot as plt

# 加载捕获文件
capture = pyshark.FileCapture('002.pcapng', display_filter='wlan.fc.type_subtype == 0x08')

# 初始化SSID和信号强度的列表
ssids = []
signal_strengths = []

# 遍历数据包
for packet in capture:
    try:
        ssid = packet.layers[3].wlan_ssid
        signal_strength = int(packet.layers[3].wlan_radio_signal_dbm)
        ssids.append(ssid)
        signal_strengths.append(signal_strength)
    except AttributeError:
        continue

# 创建数据框
df = pd.DataFrame({'SSID': ssids, 'Signal Strength (dBm)': signal_strengths})

# 计算每个SSID的平均信号强度
avg_signal_strengths = df.groupby('SSID').mean().reset_index()

# 绘图
plt.figure(figsize=(10, 5))
plt.bar(avg_signal_strengths['SSID'], avg_signal_strengths['Signal Strength (dBm)'], color='blue')
plt.xlabel('SSID')
plt.ylabel(' (dBm)')
plt.title('')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
