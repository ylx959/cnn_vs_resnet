import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import CLASSES, ROOT

RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = ROOT / 'image'

# 顏色跟著模型走，每張圖都一樣
COLORS = {'cnn': '#2a78d6', 'plain': '#eb6834', 'resnet': '#1baf7a'}
LABELS = {'cnn': 'BasicCNN', 'plain': 'PlainCNN-20', 'resnet': 'ResNet-20'}


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()


def plot_accuracy(metrics, names):
    plt.figure(figsize=(7, 3.2))
    for i, name in enumerate(names):
        mean = metrics['summary'][name]['test_acc_mean']
        std = metrics['summary'][name]['test_acc_std']
        plt.barh(i, mean, height=0.5, color=COLORS[name])
        plt.text(mean + 1, i, f"{mean:.2f}% ± {std:.2f}", va='center')

        # 黑點是每個 seed 的結果
        seeds = [r['test_acc'] for r in metrics['runs'].values() if r['model'] == name]
        plt.scatter(seeds, [i] * len(seeds), s=24, color='black', zorder=3)

    plt.yticks(range(len(names)), [LABELS[n] for n in names])
    plt.gca().invert_yaxis()
    plt.xlim(0, 115)
    plt.xlabel('Test Accuracy (%), dots = individual seeds')
    plt.title('Test Accuracy')
    save_figure('accuracy.png')


def plot_speed(metrics, names):
    devices = list(metrics['summary'][names[0]]['speed'])
    panels = [('latency_p50_ms', 'Latency, batch 1 (ms)'),
              ('throughput_img_s', 'Throughput, batch 256 (images/s)')]

    plt.figure(figsize=(9, 5))
    position = 1
    for key, title in panels:
        for device in devices:
            plt.subplot(len(panels), len(devices), position)
            position += 1
            values = [metrics['summary'][n]['speed'][device][key] for n in names]
            plt.barh(range(len(names)), values, height=0.5,
                     color=[COLORS[n] for n in names])
            for i, value in enumerate(values):
                plt.text(value, i, f"  {value:,.2f}" if value < 100 else f"  {value:,.0f}", va='center')
            plt.yticks(range(len(names)), [LABELS[n] for n in names])
            plt.gca().invert_yaxis()
            plt.xlim(0, max(values) * 1.3)
            plt.title(f"{title} - {device.upper()}", fontsize=10)
    save_figure('speed.png')


def plot_curves(names):
    plt.figure(figsize=(14, 5))
    panels = [('train_loss', 'Train Loss', 'Loss'), ('val_loss', 'Val Loss', 'Loss'),
              ('val_acc', 'Val Accuracy', 'Accuracy (%)')]

    for position, (key, title, ylabel) in enumerate(panels, start=1):
        plt.subplot(1, 3, position)
        for name in names:
            # 把同一個模型的所有 seed 取平均
            histories = []
            for path in sorted((RESULTS_DIR / 'runs').glob(f"{name}_seed*.json")):
                with open(path) as f:
                    histories.append(json.load(f)[key])
            mean = [sum(values) / len(values) for values in zip(*histories)]
            plt.plot(range(1, len(mean) + 1), mean, color=COLORS[name],
                     linewidth=2, label=LABELS[name])
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    save_figure('curves.png')


def plot_per_class(metrics, names):
    plt.figure(figsize=(10, 4))
    width = 0.8 / len(names)

    for i, name in enumerate(names):
        runs = [r for r in metrics['runs'].values() if r['model'] == name]
        values = [sum(r['per_class_acc'][c] for r in runs) / len(runs) for c in CLASSES]
        positions = [x + i * width for x in range(len(CLASSES))]
        plt.bar(positions, values, width * 0.9, color=COLORS[name], label=LABELS[name])

    plt.xticks([x + width * (len(names) - 1) / 2 for x in range(len(CLASSES))], CLASSES, rotation=25)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend(loc='lower right')
    plt.title('Per-class Test Accuracy')
    save_figure('per_class.png')


if __name__ == '__main__':
    metrics_path = RESULTS_DIR / 'metrics.json'
    if not metrics_path.exists():
        raise SystemExit(f"找不到 {metrics_path}，請先訓練模型，再執行 uv run python code/evalute.py")

    with open(metrics_path) as f:
        metrics = json.load(f)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    # 只畫實際有結果的模型
    names = [name for name in COLORS if name in metrics['summary']]

    plot_accuracy(metrics, names)
    plot_speed(metrics, names)
    plot_curves(names)
    plot_per_class(metrics, names)
    print(f"圖表已輸出到 {FIGURES_DIR}")
