from scapy.all import PcapReader
import matplotlib.pyplot as plt


pcapng_file = '003.pcapng'


ssids = []
signal_strengths = []

packets = PcapReader(pcapng_file)


for pkt in packets:
    if pkt.haslayer("Dot11Beacon"):
        ssid = pkt.info.decode()
        signal_strength = pkt.dBm_AntSignal
        ssids.append(ssid)
        signal_strengths.append(signal_strength)


plt.figure(figsize=(12, 6))
plt.bar(ssids, signal_strengths, color='skyblue')
plt.xlabel('SSID')
plt.ylabel('Signal Strength (dBm)')
plt.title('Signal Strength of Wi-Fi Networks')


for i, strength in enumerate(signal_strengths):
    plt.text(i, strength + 1, f'{strength} dBm', ha='center')

plt.xticks(rotation=45, ha='right')
plt.tight_layout(pad=2.0)


plt.show()
