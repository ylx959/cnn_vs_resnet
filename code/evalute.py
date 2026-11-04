import json
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from data import CLASSES, ROOT, get_dataloaders
from train import MODEL_DIR, RUNS_DIR, build_model, get_device, synchronize

OUTPUT_PATH = ROOT / 'results' / 'metrics.json'
LATENCY_RUNS = 100
THROUGHPUT_RUNS = 10
THROUGHPUT_BATCH = 256
ROUNDS = 5


def test_model(model, testloader, device):
    model.eval()
    # confusion[i][j]：實際類別 i 被預測成類別 j 的張數
    confusion = torch.zeros(10, 10, dtype=torch.long)

    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs.to(device))
            predicted = outputs.argmax(1).cpu()
            for label, pred in zip(labels, predicted):
                confusion[label, pred] += 1

    accuracy = 100 * confusion.diag().sum().item() / confusion.sum().item()
    per_class = 100 * confusion.diag().float() / confusion.sum(1)
    return {
        'test_acc': accuracy,
        'per_class_acc': dict(zip(CLASSES, per_class.tolist())),
        'confusion': confusion.tolist(),
    }


def measure_time(model, inputs, device, runs):
    # 回傳每次前向傳播的耗時（ms）
    times = []
    with torch.no_grad():
        for _ in range(runs):
            synchronize(device)
            start_time = time.perf_counter()
            model(inputs)
            synchronize(device)
            times.append((time.perf_counter() - start_time) * 1000)
    return times


def measure_speed(models, device):
    single_image = torch.randn(1, 3, 32, 32).to(device)
    big_batch = torch.randn(THROUGHPUT_BATCH, 3, 32, 32).to(device)
    latency = {name: [] for name in models}
    batch_time = {name: [] for name in models}

    for model in models.values():
        model.to(device).eval()
        # Warm up
        measure_time(model, single_image, device, 20)
        measure_time(model, big_batch, device, 3)

    # 各模型輪流量測，避免機器升溫降頻只影響到後面量的模型
    for _ in range(ROUNDS):
        for name, model in models.items():
            latency[name] += measure_time(model, single_image, device, LATENCY_RUNS)
            batch_time[name] += measure_time(model, big_batch, device, THROUGHPUT_RUNS)

    speed = {}
    for name in models:
        sorted_latency = sorted(latency[name])
        speed[name] = {
            'latency_p50_ms': statistics.median(sorted_latency),
            'latency_p95_ms': sorted_latency[int(len(sorted_latency) * 0.95)],
            'throughput_img_s': THROUGHPUT_BATCH / (statistics.median(batch_time[name]) / 1000),
        }
    return speed


def count_macs(model):
    # FlopCounterMode 把一次乘加算成 2 FLOPs，除以 2 得到 MACs
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(torch.randn(1, 3, 32, 32))
    return counter.get_total_flops() // 2


if __name__ == '__main__':
    checkpoints = sorted(MODEL_DIR.glob('*.pth'))
    if not checkpoints:
        raise SystemExit(f"{MODEL_DIR} 裡沒有 .pth 檔，請先執行 uv run python code/train.py")

    device = get_device()
    print(f"Using device: {device}")
    _, _, testloader = get_dataloaders(batch_size=256)

    # 測試集準確率：每個 seed 各算一次
    runs = {}
    models = {}
    for path in checkpoints:
        run_name = path.stem
        model_name = run_name.split('_seed')[0]

        model = build_model(model_name)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.to(device)

        with open(RUNS_DIR / f"{run_name}.json") as f:
            history = json.load(f)

        runs[run_name] = test_model(model, testloader, device)
        runs[run_name]['model'] = model_name
        runs[run_name]['best_val_acc'] = max(history['val_acc'])
        # 第一個 epoch 含啟動成本，不計入
        runs[run_name]['sec_per_epoch'] = statistics.median(history['seconds'][1:] or history['seconds'])
        runs[run_name]['train_seconds'] = sum(history['seconds'])
        print(f"{run_name}: Test Acc {runs[run_name]['test_acc']:.2f}%", flush=True)

        # 速度只跟架構有關，每個架構留一個模型來量
        models.setdefault(model_name, model)

    # 速度：MPS 和 CPU 都量（請在機器閒置時執行）
    speed = {device.type: measure_speed(models, device)}
    speed['cpu'] = measure_speed(models, torch.device('cpu'))

    summary = {}
    for name, model in models.items():
        accuracies = [r['test_acc'] for r in runs.values() if r['model'] == name]
        params = sum(p.numel() for p in model.parameters())
        summary[name] = {
            'seeds': len(accuracies),
            'test_acc_mean': statistics.mean(accuracies),
            'test_acc_std': statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            'params': params,
            'size_mb': params * 4 / (1024 ** 2),
            'macs': count_macs(model.cpu()),
            'speed': {d: speed[d][name] for d in speed},
        }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump({'summary': summary, 'runs': runs}, f, indent=2)
    print(json.dumps(summary, indent=2))
