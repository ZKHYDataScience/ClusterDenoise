import os
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 定义程序中用到的常量参数
DEFAULT_EPS = 2000  # DBSCAN聚类的默认距离阈值
DEFAULT_MIN_SAMPLES = 1  # DBSCAN聚类的默认最小样本数
DEFAULT_STATS_OUTPUT_PATH = "data/1/featureph3"  # 默认的统计数据输出路径
Z_MAX_THRESHOLD = 2500  # Z轴最大值阈值
POINT_COUNT_THRESHOLD = 10  # 点数阈值
FIGURE_SIZE = (10, 6)  # 图表大小

def deduplicate_csv(file_path, output_path):
    """
    CSV文件去重模块
    读取包含XYZ坐标的CSV文件，删除重复的点，并将结果保存到新文件
    同时打印去重的统计信息
    """
    df = pd.read_csv(file_path, header=None, names=["x", "y", "z"])
    original_count = len(df)
    df_deduplicated = df.drop_duplicates()
    deduplicated_count = len(df_deduplicated)
    if original_count > deduplicated_count:
        print(f"文件 {file_path} 存在重复的XYZ点，重复数: {original_count - deduplicated_count}")
    df_deduplicated.to_csv(output_path, index=False, header=False)
    print(f"去重后的数据保存到: {output_path}")

def cluster_csv(file_path, output_path, eps=DEFAULT_EPS, min_samples=DEFAULT_MIN_SAMPLES):
    """
    CSV文件聚类模块
    使用DBSCAN算法对XYZ坐标点进行聚类
    将聚类结果（包含簇标签）保存到新文件
    """
    df = pd.read_csv(file_path, header=None, names=["x", "y", "z"])
    data = df[["x", "y", "z"]].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    df["cluster"] = labels
    df.to_csv(output_path, index=False, header=False)
    print(f"聚类后的数据保存到: {output_path}")

def print_cluster_count(file_path):
    """
    聚类统计模块
    读取带有聚类标签的CSV文件
    统计并打印有效聚类的数量（不包括噪声点，即标签为-1的点）
    """
    df = pd.read_csv(file_path, header=None, names=["x", "y", "z", "cluster"])
    cluster_labels = df["cluster"]
    unique_clusters = cluster_labels[cluster_labels != -1].nunique()
    print(f"文件 {file_path} 中的聚类数量: {unique_clusters}")

def calculate_and_plot_cluster_stats(file_path, output_path):
    """
    聚类分析与可视化模块
    1. 计算每个簇的统计信息（点数、Z值范围）
    2. 将统计信息保存为CSV文件
    3. 生成簇的统计可视化图表
    """
    df = pd.read_csv(file_path, header=None, names=["x", "y", "z", "cluster"])
    df_filtered = df[df["cluster"] != -1]
    
    # 计算每个簇的统计信息
    cluster_stats = df_filtered.groupby("cluster").agg(
        point_count=("z", "size"),
        z_max=("z", "max"),
        z_min=("z", "min")
    ).reset_index()
    
    # 保存统计数据
    output_file = os.path.join(output_path, os.path.basename(file_path).replace(".csv", "_stats.csv"))
    cluster_stats.to_csv(output_file, index=False)
    print(f"统计数据保存到: {output_file}")
    
    # 生成统计可视化图表
    colors = plt.cm.get_cmap('tab20', len(cluster_stats))
    plt.figure(figsize=FIGURE_SIZE)
    
    for idx, row in cluster_stats.iterrows():
        color = colors(idx)
        plt.plot([row["point_count"], row["point_count"]], 
                [row["z_min"], row["z_max"]], 
                'o-', 
                color=color, 
                label=f'Cluster {int(row["cluster"])}')
    
    plt.xlabel("Point Count (Number of Points in Cluster)")
    plt.ylabel("Z Value (Min and Max)")
    plt.title(f"Z Min and Max for Clusters in {os.path.basename(file_path)}")
    plt.legend()
    
    # 保存图表
    plot_file = os.path.join(output_path, os.path.basename(file_path).replace(".csv", "_2d_plot.png"))
    plt.savefig(plot_file)
    plt.close()
    print(f"统计图保存到: {plot_file}")

def remove_clusters(file_path, output_path):
    """
    簇过滤模块
    删除满足特定条件的簇：
    1. Z轴最大值大于阈值
    2. 点数小于阈值
    将过滤后的结果保存到新文件
    """
    df = pd.read_csv(file_path, header=None, names=["x", "y", "z", "cluster"])
    df_filtered = df[df["cluster"] != -1]
    cluster_stats = df_filtered.groupby("cluster").agg(
        point_count=("z", "size"),
        z_max=("z", "max")
    ).reset_index()
    clusters_to_remove = cluster_stats[
        (cluster_stats["z_max"] > Z_MAX_THRESHOLD) & 
        (cluster_stats["point_count"] < POINT_COUNT_THRESHOLD)
    ]["cluster"].tolist()
    df_filtered = df[~df["cluster"].isin(clusters_to_remove)]
    output_file = os.path.join(output_path, os.path.basename(file_path))
    df_filtered.to_csv(output_file, index=False, header=False)
    print(f"满足条件的簇已删除，处理后的数据保存到: {output_file}")

def main():
    """
    主函数：处理命令行参数并按顺序执行数据处理流程
    1. 文件去重
    2. 数据聚类
    3. 聚类统计（可选）
    4. 生成统计图表（可选）
    5. 过滤特定簇
    """
    parser = argparse.ArgumentParser(description="CSV文件处理和聚类脚本")
    
    # 必须的参数
    parser.add_argument("--input_dir", type=str, required=True, help="输入的CSV文件夹路径")
    parser.add_argument("--output_deduplication", type=str, required=True, help="去重后CSV文件保存路径")
    parser.add_argument("--output_cluster", type=str, required=True, help="聚类后CSV文件保存路径")
    parser.add_argument("--output_filtered", type=str, required=True, help="删除簇后保存路径")
    
    # 可选的参数
    parser.add_argument("--check_clusters", type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help="是否打印聚类数量 (默认: True)")
    parser.add_argument("--generate_stats", type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help="是否生成统计图和CSV文件 (默认: True)")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, 
                        help=f"DBSCAN中的eps参数 (默认: {DEFAULT_EPS})")
    parser.add_argument("--min_samples", type=int, default=DEFAULT_MIN_SAMPLES, 
                        help=f"DBSCAN中的min_samples参数 (默认: {DEFAULT_MIN_SAMPLES})")

    args = parser.parse_args()

    # 创建必要的输出目录
    os.makedirs(args.output_deduplication, exist_ok=True)
    os.makedirs(args.output_cluster, exist_ok=True)
    os.makedirs(args.output_filtered, exist_ok=True)

    # 设置统计输出路径
    if args.generate_stats:
        os.makedirs(DEFAULT_STATS_OUTPUT_PATH, exist_ok=True)

    # 执行数据处理流程
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".csv"):
            # 步骤1：去重
            raw_file = os.path.join(args.input_dir, filename)
            dedup_file = os.path.join(args.output_deduplication, filename)
            deduplicate_csv(raw_file, dedup_file)

            # 步骤2：聚类
            cluster_file = os.path.join(args.output_cluster, filename)
            cluster_csv(dedup_file, cluster_file, eps=args.eps, min_samples=args.min_samples)

            # 步骤3：查看聚类情况（可选）
            if args.check_clusters:
                print_cluster_count(cluster_file)

            # 步骤4：生成统计图和CSV（可选）
            if args.generate_stats:
                calculate_and_plot_cluster_stats(cluster_file, DEFAULT_STATS_OUTPUT_PATH)

            # 步骤5：删除满足条件的簇
            remove_clusters(cluster_file, args.output_filtered)

if __name__ == "__main__":
    main()