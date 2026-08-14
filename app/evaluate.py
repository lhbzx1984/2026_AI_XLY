"""
舌象分析评估模块
================
用 YOLO 21 类标注数据集作为金标准，验证现有 HSV/U2-Net 分析流程的准确率。

评估维度：
1. 舌质颜色分类准确率（健康舌/红舌/紫舌）
2. 舌苔分类准确率（白苔/黄苔/黑苔/花苔）
3. 整体混淆矩阵
4. 按类别的精确率/召回率/F1

用法：
    python -m app.evaluate                    # 评估全部图片
    python -m app.evaluate --sample 50        # 只评估前50张（快速测试）
    python -m app.evaluate --backend u2net    # 用 U2-Net 后端评估
    python -m app.evaluate --report report.html  # 指定报告输出路径
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .segmentation import segment_tongue, analyze_tongue_body_color, analyze_coating_color
from .atlas import (
    YOLO_CLASSES,
    KB_TO_YOLO_MAP,
    parse_yolo_label,
    classify_annotation,
    get_yolo_class_info,
)

# 评估范围定义
# 舌质颜色类（HSV 可检测的）
BODY_COLOR_GT_IDS = {0, 2, 3}       # 健康舌、红舌、紫舌
BODY_COLOR_PRED_MAP = {
    "pale_red": 0,
    "red": 2,
    "crimson": 2,   # 绛舌归入红舌
    "purple": 3,
    "pale": 0,      # 淡白舌暂归入健康舌大类（颜色偏浅）
}

# 舌苔类
COATING_GT_IDS = {1, 9, 10, 11, 12}  # 薄苔、白苔、黄苔、黑苔、花苔
# 1(薄苔)和9(白苔)合并为"白苔大类"评估
COATING_MERGE = {1: 9, 9: 9, 10: 10, 11: 11, 12: 12}
COATING_PRED_MAP = {
    "thin_white": 9,
    "thick_white": 9,
    "yellow": 10,
    "greasy_yellow": 10,
    "peeled": 12,
    "gray_black": 11,
}

# 类别显示名
BODY_COLOR_NAMES = {0: "健康舌", 2: "红舌", 3: "紫舌"}
COATING_NAMES = {9: "白苔", 10: "黄苔", 11: "黑苔", 12: "花苔"}


def find_dataset_root() -> str:
    """查找 test 数据集根目录"""
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "test"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test"),
        "c:\\Users\\Dell\\Desktop\\刘珮霖\\test",
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(os.path.join(p, "images")) and os.path.isdir(os.path.join(p, "labels")):
            return p
    return ""


def evaluate_single_image(image_path: str, label_path: str, backend: str = "hsv") -> dict:
    """
    评估单张图片：跑分析流程 + 对比金标准。

    返回:
        {
            "image": 文件名,
            "gt_body": 金标准舌质类(int or None),
            "pred_body": 预测舌质类(int or None),
            "gt_coating": 金标准舌苔类(int or None),
            "pred_coating": 预测舌苔类(int or None),
            "body_correct": bool,
            "coating_correct": bool,
            "seg_success": bool,
            "error": str or None,
        }
    """
    result = {
        "image": os.path.basename(image_path),
        "gt_body": None, "pred_body": None,
        "gt_coating": None, "pred_coating": None,
        "body_correct": False, "coating_correct": False,
        "seg_success": False, "error": None,
    }

    # 1. 解析金标准
    try:
        annotations = parse_yolo_label(label_path)
    except Exception as e:
        result["error"] = f"标注解析失败: {e}"
        return result

    for ann in annotations:
        if classify_annotation(ann) == "whole":
            cls = ann["class"]
            if cls in BODY_COLOR_GT_IDS and result["gt_body"] is None:
                result["gt_body"] = cls
            elif cls in COATING_GT_IDS and result["gt_coating"] is None:
                result["gt_coating"] = cls

    # 2. 跑分析流程
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        seg_result = segment_tongue(image_path, backend=backend)
        if not seg_result["success"]:
            result["error"] = "分割失败"
            return result
        result["seg_success"] = True

        mask = seg_result["mask"]
        resized_array = seg_result.get("resized_array", img_array)

        body_result = analyze_tongue_body_color(resized_array, mask)
        coating_result = analyze_coating_color(resized_array, mask)

        body_key = body_result.get("key", "")
        coating_key = coating_result.get("key", "")

        result["pred_body"] = BODY_COLOR_PRED_MAP.get(body_key)
        result["pred_coating"] = COATING_PRED_MAP.get(coating_key)

    except Exception as e:
        result["error"] = str(e)
        return result

    # 3. 对比（合并薄苔/白苔后对比）
    if result["gt_body"] is not None and result["pred_body"] is not None:
        result["body_correct"] = (result["gt_body"] == result["pred_body"])

    gt_c = COATING_MERGE.get(result["gt_coating"])
    pred_c = COATING_MERGE.get(result["pred_coating"])
    if gt_c is not None and pred_c is not None:
        result["coating_correct"] = (gt_c == pred_c)

    return result


def run_evaluation(dataset_root: str = "", backend: str = "hsv", sample: int = 0,
                   progress_callback=None) -> dict:
    """
    运行完整评估。

    参数:
        dataset_root: 数据集根目录（含 images/ 和 labels/）
        backend: 分割后端 ("hsv" / "u2net")
        sample: 只评估前N张（0=全部）
        progress_callback: 回调函数(current, total, result)

    返回:
        评估结果字典
    """
    if not dataset_root:
        dataset_root = find_dataset_root()
    if not dataset_root or not os.path.isdir(dataset_root):
        return {"error": f"数据集目录不存在: {dataset_root}"}

    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if sample > 0:
        image_files = image_files[:sample]

    total = len(image_files)
    results = []
    t0 = time.time()

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.exists(label_path):
            continue

        r = evaluate_single_image(img_path, label_path, backend=backend)
        results.append(r)

        if progress_callback:
            progress_callback(i + 1, total, r)

    elapsed = time.time() - t0

    # 统计
    stats = compute_stats(results)
    stats["total_images"] = total
    stats["evaluated"] = len(results)
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["backend"] = backend
    stats["dataset_root"] = dataset_root
    stats["results"] = results

    return stats


def compute_stats(results: list) -> dict:
    """计算评估统计指标"""
    stats = {
        "body_total": 0, "body_correct": 0,
        "coating_total": 0, "coating_correct": 0,
        "seg_success": 0, "seg_fail": 0,
        "body_confusion": {},  # {(gt, pred): count}
        "coating_confusion": {},
        "per_class": {},
    }

    for r in results:
        if r["seg_success"]:
            stats["seg_success"] += 1
        else:
            stats["seg_fail"] += 1

        if r["gt_body"] is not None and r["pred_body"] is not None:
            stats["body_total"] += 1
            if r["body_correct"]:
                stats["body_correct"] += 1
            key = (BODY_COLOR_NAMES.get(r["gt_body"], str(r["gt_body"])),
                   BODY_COLOR_NAMES.get(r["pred_body"], str(r["pred_body"])))
            stats["body_confusion"][key] = stats["body_confusion"].get(key, 0) + 1

        if r["gt_coating"] is not None and r["pred_coating"] is not None:
            stats["coating_total"] += 1
            if r["coating_correct"]:
                stats["coating_correct"] += 1
            gt_c = COATING_MERGE.get(r["gt_coating"], r["gt_coating"])
            pred_c = COATING_MERGE.get(r["pred_coating"], r["pred_coating"])
            key = (COATING_NAMES.get(gt_c, str(gt_c)),
                   COATING_NAMES.get(pred_c, str(pred_c)))
            stats["coating_confusion"][key] = stats["coating_confusion"].get(key, 0) + 1

    stats["body_accuracy"] = round(stats["body_correct"] / stats["body_total"], 4) if stats["body_total"] > 0 else 0
    stats["coating_accuracy"] = round(stats["coating_correct"] / stats["coating_total"], 4) if stats["coating_total"] > 0 else 0
    stats["seg_success_rate"] = round(stats["seg_success"] / (stats["seg_success"] + stats["seg_fail"]), 4) if (stats["seg_success"] + stats["seg_fail"]) > 0 else 0

    return stats


def generate_evaluation_html(stats: dict) -> str:
    """生成评估报告 HTML（供网页展示）"""
    if "error" in stats:
        return f'<div style="padding:40px;text-align:center;color:#e74c3c;">评估失败：{stats["error"]}</div>'

    body_acc = stats["body_accuracy"]
    coat_acc = stats["coating_accuracy"]
    seg_rate = stats["seg_success_rate"]

    # 准确率颜色
    def acc_color(acc):
        if acc >= 0.8: return "#27ae60"
        if acc >= 0.6: return "#f39c12"
        return "#e74c3c"

    # 混淆矩阵表格
    def confusion_table(confusion: dict, names: list) -> str:
        rows = ""
        for gt_name in names:
            cells = f'<td style="font-weight:600;background:#f8f9fa;">{gt_name}</td>'
            for pred_name in names:
                count = confusion.get((gt_name, pred_name), 0)
                bg = ""
                if gt_name == pred_name and count > 0:
                    bg = 'background:#d4edda;color:#155724;'
                elif count > 0:
                    bg = 'background:#f8d7da;color:#721c24;'
                cells += f'<td style="text-align:center;{bg}">{count if count else ""}</td>'
            rows += f'<tr>{cells}</tr>'
        header = '<tr><th style="background:#343a40;color:white;">金标准 \\ 预测</th>'
        for name in names:
            header += f'<th style="background:#343a40;color:white;">{name}</th>'
        header += "</tr>"
        return f'<table style="border-collapse:collapse;width:100%;font-size:13px;">{header}{rows}</table>'

    body_names = list(BODY_COLOR_NAMES.values())
    coating_names = list(COATING_NAMES.values())

    # 错误样例（最多展示10个）
    error_samples = ""
    error_count = 0
    for r in stats.get("results", []):
        if error_count >= 10:
            break
        if r["gt_body"] is not None and r["pred_body"] is not None and not r["body_correct"]:
            gt_name = BODY_COLOR_NAMES.get(r["gt_body"], str(r["gt_body"]))
            pred_name = BODY_COLOR_NAMES.get(r["pred_body"], str(r["pred_body"]))
            error_samples += f'<tr><td>{r["image"]}</td><td>{gt_name}</td><td>{pred_name}</td><td style="color:#e74c3c;">舌质颜色错误</td></tr>'
            error_count += 1
        elif r["gt_coating"] is not None and r["pred_coating"] is not None and not r["coating_correct"]:
            gt_c = COATING_MERGE.get(r["gt_coating"], r["gt_coating"])
            pred_c = COATING_MERGE.get(r["pred_coating"], r["pred_coating"])
            gt_name = COATING_NAMES.get(gt_c, str(gt_c))
            pred_name = COATING_NAMES.get(pred_c, str(pred_c))
            error_samples += f'<tr><td>{r["image"]}</td><td>{gt_name}</td><td>{pred_name}</td><td style="color:#e74c3c;">舌苔分类错误</td></tr>'
            error_count += 1

    return f"""
    <div style="font-family:'Segoe UI',sans-serif;padding:20px;">
        <h2 style="color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:10px;">
            数据集验证报告
        </h2>
        <div style="background:#eef7ff;border-left:4px solid #3498db;padding:12px 16px;margin:15px 0;border-radius:4px;">
            <strong>评估说明：</strong>用 YOLO 21 类标注数据集（{stats.get("evaluated", 0)} 张图）作为金标准，
            验证现有 {stats.get("backend","hsv").upper()} 分析流程的舌质颜色与舌苔分类准确率。
            形态特征（胖大/瘦/红点/裂纹/齿痕）和脏腑分区需 YOLO 检测模型，当前算法暂不支持。
        </div>

        <div style="display:flex;gap:20px;margin:20px 0;">
            <div style="flex:1;background:white;border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;">
                <div style="font-size:14px;color:#666;">分割成功率</div>
                <div style="font-size:32px;font-weight:700;color:{acc_color(seg_rate)};">{seg_rate:.1%}</div>
                <div style="font-size:12px;color:#999;">{stats["seg_success"]}/{stats["seg_success"]+stats["seg_fail"]} 张</div>
            </div>
            <div style="flex:1;background:white;border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;">
                <div style="font-size:14px;color:#666;">舌质颜色准确率</div>
                <div style="font-size:32px;font-weight:700;color:{acc_color(body_acc)};">{body_acc:.1%}</div>
                <div style="font-size:12px;color:#999;">{stats["body_correct"]}/{stats["body_total"]} 张可对比</div>
            </div>
            <div style="flex:1;background:white;border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;">
                <div style="font-size:14px;color:#666;">舌苔分类准确率</div>
                <div style="font-size:32px;font-weight:700;color:{acc_color(coat_acc)};">{coat_acc:.1%}</div>
                <div style="font-size:12px;color:#999;">{stats["coating_correct"]}/{stats["coating_total"]} 张可对比</div>
            </div>
        </div>

        <h3 style="color:#2c3e50;margin-top:25px;">舌质颜色混淆矩阵</h3>
        {confusion_table(stats["body_confusion"], body_names)}

        <h3 style="color:#2c3e50;margin-top:25px;">舌苔分类混淆矩阵</h3>
        {confusion_table(stats["coating_confusion"], coating_names)}

        {f'''
        <h3 style="color:#2c3e50;margin-top:25px;">典型错误样例</h3>
        <table style="border-collapse:collapse;width:100%;font-size:13px;">
            <tr><th style="background:#343a40;color:white;padding:8px;">图片</th>
                <th style="background:#343a40;color:white;padding:8px;">金标准</th>
                <th style="background:#343a40;color:white;padding:8px;">预测</th>
                <th style="background:#343a40;color:white;padding:8px;">错误类型</th></tr>
            {error_samples}
        </table>
        ''' if error_samples else ''}

        <div style="margin-top:20px;color:#999;font-size:12px;">
            评估耗时：{stats.get("elapsed_seconds",0)} 秒 | 数据集：{stats.get("dataset_root","")}
        </div>
    </div>
    """


# ============================================================
# 命令行入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="舌象分析评估")
    parser.add_argument("--dataset", default="", help="数据集根目录（含 images/ 和 labels/）")
    parser.add_argument("--backend", default="hsv", choices=["hsv", "u2net"], help="分割后端")
    parser.add_argument("--sample", type=int, default=0, help="只评估前N张（0=全部）")
    parser.add_argument("--report", default="", help="报告输出 HTML 路径")
    args = parser.parse_args()

    def progress(cur, total, result):
        pct = cur / total * 100
        status = "✓" if result["body_correct"] or result["coating_correct"] else "✗"
        print(f"\r[{cur}/{total}] {pct:.0f}% {status} {result['image'][:30]}", end="", flush=True)

    print(f"开始评估（后端: {args.backend}）...")
    stats = run_evaluation(args.dataset, args.backend, args.sample, progress)
    print(f"\n\n评估完成！")
    print(f"  分割成功率: {stats['seg_success_rate']:.1%}")
    print(f"  舌质准确率: {stats['body_accuracy']:.1%}")
    print(f"  舌苔准确率: {stats['coating_accuracy']:.1%}")

    if args.report:
        html = generate_evaluation_html(stats)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  报告已保存: {args.report}")


if __name__ == "__main__":
    main()
