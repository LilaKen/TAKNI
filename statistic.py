import os
import numpy as np
import csv
from collections import defaultdict


def parse_folder_name(folder_name):
    parts = folder_name.split('_')
    if len(parts) < 10:
        print(f"文件夹名称格式不正确: {folder_name}")
        return None

    model = "_".join(parts[0:3])  # 将 "cnn_features_1d" 组合在一起
    dataset = parts[3]
    source_to_target = parts[4] + "_" + parts[5] + "_" + parts[6]
    year = parts[7]
    cda_flag = parts[8].lower() == "true"  # 解析 True 或 False 标志
    method_or_algorithm = parts[9]  # 可能是 "CDA" 或 "DA"
    distance_flag = parts[10].lower() == "true"  # True 或 False
    algorithm = parts[11]  # MK-MMD、CORAL 等

    if cda_flag:
        algorithm = parts[9]
    elif distance_flag:
        algorithm = parts[11]

    # 确保 CDA 和 DA 映射为 CDAN 和 DANN
    if algorithm == "DA":
        algorithm = "DANN"
    elif algorithm == "CDA":
        algorithm = "CDAN"
    else:
        algorithm = algorithm

    # 如果算法是 "CDA+E"，则跳过
    if algorithm == "CDA+E" and cda_flag:
        # print(f"跳过文件夹: {folder_name}，因为算法是 CDA+E")
        return None

    return {
        "Model": model,
        "Dataset": dataset,
        "Source to Target": source_to_target,
        "Year": int(year),
        "Algorithm": algorithm,
        "Best": None,
        "Last": None
    }

def parse_folder_in_fft_name(folder_name):
    parts = folder_name.split('_')
    if len(parts) < 10:
        print(f"文件夹名称格式不正确: {folder_name}")
        return None

    model = "_".join(parts[0:4])  # 将 "cnn_features_1d" 组合在一起
    dataset = parts[4]
    source_to_target = parts[5] + "_" + parts[6] + "_" + parts[7]
    year = parts[8]
    cda_flag = parts[9].lower() == "true"  # 解析 True 或 False 标志
    method_or_algorithm = parts[10]  # 可能是 "CDA" 或 "DA"
    distance_flag = parts[11].lower() == "true"  # True 或 False
    algorithm = parts[12]  # MK-MMD、CORAL 等

    if cda_flag:
        algorithm = parts[10]
    elif distance_flag:
        algorithm = parts[12]

    # 确保 CDA 和 DA 映射为 CDAN 和 DANN
    if algorithm == "DA":
        algorithm = "DANN"
    elif algorithm == "CDA":
        algorithm = "CDAN"
    else:
        algorithm = algorithm

    # 如果算法是 "CDA+E"，则跳过
    if algorithm == "CDA+E" and cda_flag:
        # print(f"跳过文件夹: {folder_name}，因为算法是 CDA+E")
        return None

    return {
        "Model": model,
        "Dataset": dataset,
        "Source to Target": source_to_target,
        "Year": int(year),
        "Algorithm": algorithm,
        "Best": None,
        "Last": None
    }

def extract_score_from_filename(filename):
    try:
        parts = filename.split('-')
        if len(parts) >= 3:
            score = round(float(parts[1]) * 100, 2)  # 提取分数并乘以100转化为百分比，并保留两位小数
            return score
    except Exception as e:
        print(f"无法从文件名中提取分数: {filename}, 错误: {e}")
    return None


def parse_files_in_folder(folder_path):
    best_score = None
    last_score = None

    for filename in os.listdir(folder_path):
        if filename.endswith(".pth"):
            score = extract_score_from_filename(filename)
            if score is not None:
                if best_score is None or score > best_score:
                    best_score = score
            if filename.startswith("999"):
                last_score = score

    return {
        "Best": best_score,
        "Last": last_score
    }

def parse_all_folders(directory):
    parsed_info = []

    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            if folder.startswith("informer_features_fft_1d"):
                folder_info = parse_folder_in_fft_name(folder)
            else:
                folder_info = parse_folder_name(folder)
            if folder_info:
                max_best_score = None
                file_scores = parse_files_in_folder(folder_path)
                if max_best_score is None or file_scores["Best"] > max_best_score:
                    max_best_score = file_scores["Best"]
                folder_info.update({"Best": max_best_score, "Last": file_scores["Last"]})
                parsed_info.append(folder_info)

    return parsed_info


def calculate_max_mean_by_task(data):
    grouped_results = defaultdict(lambda: defaultdict(list))

    for folder in data:
        source_to_target = folder['Source to Target']
        algorithm = folder['Algorithm']

        if folder['Best'] is not None:
            grouped_results[source_to_target][f"{algorithm}_Best"].append(folder['Best'])
        if folder['Last'] is not None:
            grouped_results[source_to_target][f"{algorithm}_Last"].append(folder['Last'])

    results = {}
    for source_to_target, algorithms_data in grouped_results.items():
        results[source_to_target] = {}
        for key, values in algorithms_data.items():
            if values:
                max_value = round(max(values), 2)
                mean_value = round(np.mean(values), 2)
                results[source_to_target][f"{key}_Max"] = max_value
                results[source_to_target][f"{key}_Mean"] = mean_value
            else:
                results[source_to_target][f"{key}_Max"] = ''
                results[source_to_target][f"{key}_Mean"] = ''

    return results


def display_single_table(task_name, results):
    # algorithms = ['MK-MMD', 'JMMD', 'LJMMD', 'CORAL', 'DANN', 'CDAN', 'Ours']
    algorithms = ['MK-MMD', 'JMMD', 'LJMMD', 'CORAL', 'DANN', 'CDAN', 'NWD']


    print(f"\nTable for Task: {task_name}\n")

    header_row_1 = "Task  | Loc  | Basis  | AdaBN  "
    header_row_2 = "Source|       |        |         "

    for algo in algorithms:
        if algo == 'NWD':
            algo = 'Ours'
        header_row_1 += f" | {algo}      "
        header_row_2 += f" | Max | Mean"

    print(header_row_1)
    print(header_row_2)

    for source_to_target, values in results.items():
        for loc in ['Best', 'Last']:
            row = f"{source_to_target} | {loc} "
            row += " | " * 3  # Basis 和 AdaBN 留空

            for algo in algorithms:
                row += f" | {values.get(f'{algo}_{loc}_Max', '')} | {values.get(f'{algo}_{loc}_Mean', '')}"
            print(row)


def save_results_to_csv(task_name, results, output_file):
    # algorithms = ['MK-MMD', 'JMMD', 'LJMMD', 'CORAL', 'DANN', 'CDAN', 'Ours']
    algorithms = ['MK-MMD', 'JMMD', 'LJMMD', 'CORAL', 'DANN', 'CDAN', 'NWD']


    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头
        header_row_1 = ["Task", "Loc", "Basis", "AdaBN"]
        for algo in algorithms:
            header_row_1.extend([f"{algo} Max", f"{algo} Mean"])
        writer.writerow(header_row_1)

        # 写入第二行表头
        header_row_2 = ["Source", "", "", ""]
        for algo in algorithms:
            header_row_2.extend(["Max", "Mean"])
        writer.writerow(header_row_2)

        # 写入数据
        for source_to_target, values in results.items():
            for loc in ['Best', 'Last']:
                row = [source_to_target, loc, '', '']  # Basis 和 AdaBN 留空

                for algo in algorithms:
                    row.append(values.get(f'{algo}_{loc}_Max', ''))
                    row.append(values.get(f'{algo}_{loc}_Mean', ''))
                writer.writerow(row)

def parse_folders_for_multiple_paths(paths):
    """
    处理多个路径，并分别处理每个路径下的文件夹
    """
    for path in paths:
        print(f"\nProcessing directory: {path}")
        parsed_results = parse_all_folders(path)
        if parsed_results:
            task_name = os.path.basename(path)  # 获取路径名作为任务名
            task_results = calculate_max_mean_by_task(parsed_results)
            display_single_table(task_name, task_results)
            output_file = f"{task_name}.csv"
            save_results_to_csv(task_name, task_results, output_file)


# 示例使用，适用于多个路径
paths = [
    # "checkpoint/cnn/CWRU/None",  # 第一个路径
    # "checkpoint/cnn/PU/None",  # 第二个路径
    # "checkpoint/cnn/JNU/None",  # 第一个路径
    # "checkpoint/cnn/PHM/None",  # 第一个路径
    # "checkpoint/cnn/SEU/None",  # 第一个路径


    # "checkpoint/cnn/CWRU/FFT",  # 第二个路径
    # "checkpoint/cnn/PU/FFT",  # 第二个路径
    # "checkpoint/cnn/JNU/FFT",  # 第二个路径
    # "checkpoint/cnn/PHM/FFT",  # 第二个路径
    # "checkpoint/cnn/SEU/FFT",  # 第二个路径


    # "checkpoint/informer/CWRU/None",  # 第一个路径
    # "checkpoint/informer/PU/None",  # 第一个路径
    # "checkpoint/informer/JNU/None",  # 第一个路径
    # "checkpoint/informer/PHM/None",  # 第一个路径
    # "checkpoint/informer/SEU/None",  # 第一个路径
    #
    # "checkpoint/informer/CWRU/FFT",  # 第二个路径
    # "checkpoint/informer/PU/FFT",  # 第一个路径
    # "checkpoint/informer/JNU/FFT",  # 第二个路径
    # "checkpoint/informer/PHM/FFT",  # 第二个路径
    # "checkpoint/informer/SEU/FFT",  # 第二个路径


    # 继续添加其他路径...
]

# 分别处理每个路径的文件夹
parse_folders_for_multiple_paths(paths)
