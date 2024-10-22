import os
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import argparse

# 定义函数，用于去重CSV文件
def deduplicate_csv(file_path, output_path):
    df = pd.read_csv(file_path, header=None, names=["x", "y", "z"])
    original_count = len(df)
    df_deduplicated = df.drop_duplicates()
    deduplicated_count = len(df_deduplicated)
    if original_count > deduplicated_count:
        print(f"文件 {file_path} 存在重复的XYZ点，重复数: {original_count - deduplicated_count}")
    df_deduplicated.to_csv(output_path, index=False, header=False)
    print(f"去重后的数据保存到: {output_path}")

# 定义函数，对CSV文件进行DBSCAN聚类
def cluster_csv(file_path, output_path, eps=2000, min_samples=1):
    df = pd.read_csv(file_path, header=None, names=["x", "y", "z"])
    data = df[["x", "y", "z"]].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    df["cluster"] = labels
    df.to_csv(output_path, index=False, header=False)
    print(f"聚类后的数据保存到: {output_path}")

# 定义函数，读取CSV文件并打印聚类的数量
def print_cluster_count(file_path):
    df = pd.read_csv(file_path, header=None, names=["x", "y", "z", "cluster"])
    cluster_labels = df["cluster"]
    unique_clusters = cluster_labels[cluster_labels != -1].nunique()
    print(f"文件 {file_path} 中的聚类数量: {unique_clusters}")

# 定义函数，统计每个簇的数量以及Z轴的最大最小值，生成统计CSV并绘制二维图
def calculate_and_plot_cluster_stats(file_path, output_path):
    df = pd.read_csv(file_path, header=None, names=["x", "y", "z", "cluster"])
    df_filtered = df[df["cluster"] != -1]
    cluster_stats = df_filtered.groupby("cluster").agg(
        point_count=("z", "size"),
        z_max=("z", "max"),
        z_min=("z", "min")
    ).reset_index()
    
    # 保存统计数据到默认路径
    output_file = os.path.join(output_path, os.path.basename(file_path).replace(".csv", "_stats.csv"))
    cluster_stats.to_csv(output_file, index=False)
    print(f"统计数据保存到: {output_file}")
    
    # 绘制统计图
    colors = plt.cm.get_cmap('tab20', len(cluster_stats))
    plt.figure(figsize=(10, 6))
    
    for idx, row in cluster_stats.iterrows():
        color = colors(idx)
        plt.plot([row["point_count"], row["point_count"]], [row["z_min"], row["z_max"]], 'o-', color=color, label=f'Cluster {int(row["cluster"])}')
    
    plt.xlabel("Point Count (Number of Points in Cluster)")
    plt.ylabel("Z Value (Min and Max)")
    plt.title(f"Z Min and Max for Clusters in {os.path.basename(file_path)}")
    plt.legend()
    
    # 保存统计图到默认路径
    plot_file = os.path.join(output_path, os.path.basename(file_path).replace(".csv", "_2d_plot.png"))
    plt.savefig(plot_file)
    plt.close()
    print(f"统计图保存到: {plot_file}")

# 定义函数，删除满足条件的簇，并保存结果
def remove_clusters(file_path, output_path):
    df = pd.read_csv(file_path, header=None, names=["x", "y", "z", "cluster"])
    df_filtered = df[df["cluster"] != -1]
    cluster_stats = df_filtered.groupby("cluster").agg(
        point_count=("z", "size"),
        z_max=("z", "max")
    ).reset_index()
    clusters_to_remove = cluster_stats[(cluster_stats["z_max"] > 2500) & (cluster_stats["point_count"] < 10)]["cluster"].tolist()
    df_filtered = df[~df["cluster"].isin(clusters_to_remove)]
    output_file = os.path.join(output_path, os.path.basename(file_path))
    df_filtered.to_csv(output_file, index=False, header=False)
    print(f"满足条件的簇已删除，处理后的数据保存到: {output_file}")

# 主函数
def main():
    parser = argparse.ArgumentParser(description="CSV文件处理和聚类脚本")
    
    # 必须的参数
    parser.add_argument("--input_dir", type=str, required=True, help="输入的CSV文件夹路径")
    parser.add_argument("--output_deduplication", type=str, required=True, help="去重后CSV文件保存路径")
    parser.add_argument("--output_cluster", type=str, required=True, help="聚类后CSV文件保存路径")
    parser.add_argument("--output_filtered", type=str, required=True, help="删除簇后保存路径")
    
    # 可选的参数，默认值为True
    parser.add_argument("--check_clusters", type=lambda x: (str(x).lower() == 'true'), default=True, help="是否打印聚类数量 (默认: True)")
    parser.add_argument("--generate_stats", type=lambda x: (str(x).lower() == 'true'), default=True, help="是否生成统计图和CSV文件 (默认: True)")
    parser.add_argument("--eps", type=float, default=2000, help="DBSCAN中的eps参数 (默认: 2000)")
    parser.add_argument("--min_samples", type=int, default=1, help="DBSCAN中的min_samples参数 (默认: 1)")

    args = parser.parse_args()

    # 创建必要的输出目录
    os.makedirs(args.output_deduplication, exist_ok=True)
    os.makedirs(args.output_cluster, exist_ok=True)
    os.makedirs(args.output_filtered, exist_ok=True)

    # 设置默认的统计输出路径
    default_stats_output_path = "data/1/featureph3"
    if args.generate_stats:
        os.makedirs(default_stats_output_path, exist_ok=True)

    # 步骤1：去重
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".csv"):
            raw_file = os.path.join(args.input_dir, filename)
            output_file = os.path.join(args.output_deduplication, filename)
            deduplicate_csv(raw_file, output_file)

    # 步骤2：聚类
    for filename in os.listdir(args.output_deduplication):
        if filename.endswith(".csv"):
            raw_file = os.path.join(args.output_deduplication, filename)
            output_file = os.path.join(args.output_cluster, filename)
            cluster_csv(raw_file, output_file, eps=args.eps, min_samples=args.min_samples)

    # 步骤3：选择是否查看聚类情况
    if args.check_clusters:
        for filename in os.listdir(args.output_cluster):
            if filename.endswith(".csv"):
                file_path = os.path.join(args.output_cluster, filename)
                print_cluster_count(file_path)

    # 步骤4：选择是否生成统计图和CSV
    if args.generate_stats:
        for filename in os.listdir(args.output_cluster):
            if filename.endswith(".csv"):
                file_path = os.path.join(args.output_cluster, filename)
                calculate_and_plot_cluster_stats(file_path, default_stats_output_path)

    # 步骤5：删除满足条件的簇
    for filename in os.listdir(args.output_cluster):
        if filename.endswith(".csv"):
            file_path = os.path.join(args.output_cluster, filename)
            remove_clusters(file_path, args.output_filtered)

if __name__ == "__main__":
    main()
