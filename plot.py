import argparse

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--cutlayer', type=str)
parser.add_argument('--data', type=str)

arg = parser.parse_args()

model = arg.model
data = arg.data
cutlayer = arg.cutlayer

fontsize = 22
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.titlesize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = 17
plt.rcParams['figure.titlesize'] = fontsize

method = ['ms', 'vq', 'qq', 'vs', 'fpq', 'rts']
labels = ['MS', 'VQ', 'QQ', 'VS', 'FPQ', 'RTS']

baseline_acc1 = np.load(f'baseline/{model}_{data}_baseline_acc1.npy')
baseline_loss = np.load(f'baseline/{model}_{data}_baseline_loss.npy')

acc1_dict = {i:np.load(f'result/{model}_{cutlayer}_{data}_{i}_acc1.npy') for i in method}
loss_dict = {i:np.load(f'result/{model}_{cutlayer}_{data}_{i}_loss.npy') for i in method}
for i, v in acc1_dict.items():
    for j in range(len(v)):
        acc1_dict[i][j] *= 100

fig, ax = plt.subplots(figsize=(6, 4.6), dpi=600, tight_layout=True)

colors = [
    '#1f77b4',  # 蓝色 (Tableau Blue)
    '#ff7f0e',  # 橙色 (Tableau Orange)
    '#2ca02c',  # 绿色 (Tableau Green)
    '#d62728',  # 红色 (Tableau Red)
    '#9467bd',  # 紫色 (Tableau Purple)
    '#8c564b'   # 棕色 (Tableau Brown)
]

line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]

for idx, (method_name, loss_values) in enumerate(loss_dict.items()):
    ax.plot(range(1, 41),
            loss_values[::5], 
            label=labels[idx],
            color=colors[idx],
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=2,
            alpha=0.8)
plt.xticks([0, 10, 20, 30, 40])


baseline_min = baseline_loss.min()
ax.axhline(y=baseline_min, 
           color='k', 
           linestyle='--', 
           linewidth=1.5,
           alpha=0.7)
ax.text(0.995, baseline_min*1.05, 'Baseline', 
        ha='right', va='bottom',
        transform=ax.get_yaxis_transform(),
        )

ax.set_xlabel('Epoch (×5)')
ax.set_ylabel('Training Loss')
ax.tick_params(axis='both', which='major')

ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
ax.legend(frameon=True, loc='upper right')

plt.savefig(f'img/{model}_{data}_{cutlayer}_loss.pdf', dpi=600, format='pdf', pad_inches=0.02, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(6, 4.6), dpi=600, tight_layout=True)
for idx, (method_name, acc1_values) in enumerate(acc1_dict.items()):
    ax.plot(range(1, 41),
            acc1_values[::5], 
            label=labels[idx],
            color=colors[idx],
            linestyle=line_styles[idx % len(line_styles)],
            linewidth=2,
            alpha=0.8)
plt.xticks([0, 10, 20, 30, 40])
plt.yticks([20, 40, 60, 80])


baseline_max = baseline_acc1.max() * 100
ax.axhline(y=baseline_max, 
           color='k', 
           linestyle='--', 
           linewidth=1.5,
           alpha=0.7)
ax.text(0.25, baseline_max*0.995, 'Baseline', 
        ha='right', va='top',
        transform=ax.get_yaxis_transform(),
        )

ax.set_xlabel('Epoch (×5)')
ax.set_ylabel('Test Accuracy(%)')
ax.tick_params(axis='both', which='major')

ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
ax.legend(frameon=True, loc='lower right')

plt.savefig(f'img/{model}_{data}_{cutlayer}_acc.pdf', dpi=600, format='pdf', pad_inches=0.02, bbox_inches='tight')