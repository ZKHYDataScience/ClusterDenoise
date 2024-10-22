
# CSV 数据去重与聚类项目

本项目提供了一个 Python 脚本 (`run.py`)，用于处理 CSV 文件，执行以下步骤：
1. 去除重复的 XYZ 数据点；
2. 使用 DBSCAN 进行聚类；
3. 可选择打印聚类数量；
4. 可选择生成统计摘要和绘制簇的 Z 轴值统计图；
5. 根据指定条件筛选并删除某些簇。

该脚本适用于包含 3D 坐标点 (x, y, z) 的 CSV 数据文件。

## 功能

1. **数据去重**：删除 CSV 文件中重复的 (x, y, z) 坐标点。
2. **DBSCAN 聚类**：对去重后的数据应用 DBSCAN 聚类算法。
3. **聚类数量检查**：可选择打印聚类数量（除噪声点外的簇）。
4. **统计摘要与绘图**：可选择生成每个簇的统计信息并绘制 Z 轴最小值和最大值的二维图。
5. **簇筛选**：根据条件（例如 `z_max > 2500` 且点数小于10）筛选并删除指定簇。

## 使用说明

运行该脚本的基本命令如下：

```bash
python run.py --input_dir <输入CSV文件夹路径> --output_deduplication <去重后数据保存路径> --output_cluster <聚类后数据保存路径> --output_filtered <筛选后数据保存路径> [可选参数]
```

### 必选参数:
- `--input_dir`: 包含输入 CSV 文件的目录路径。
- `--output_deduplication`: 去重后的 CSV 文件保存的目录路径。
- `--output_cluster`: 聚类后的 CSV 文件保存的目录路径。
- `--output_filtered`: 筛选后 CSV 文件保存的目录路径。

### 可选参数:
- `--check_clusters`: 是否打印每个文件的聚类数量 (默认: `True`)。
- `--generate_stats`: 是否生成统计信息和簇的二维图 (默认: `True`)。
- `--eps`: DBSCAN 算法中的 `eps` 参数，控制样本之间的最大距离 (默认: `2000`)。
- `--min_samples`: DBSCAN 算法中的 `min_samples` 参数，控制形成簇所需的最少点数 (默认: `1`)。

### 示例命令:

```bash
python run.py --input_dir data/raw --output_deduplication data/Deduplication1 --output_cluster data/cluster2 --output_filtered data/1/deletedata4 --check_clusters True --generate_stats True --eps 2500 --min_samples 5
```

### 处理步骤说明:

1. **数据去重**: 
    - 从 `--input_dir` 指定的目录中读取 CSV 文件，去除重复的 (x, y, z) 数据点，结果保存到 `--output_deduplication` 指定的目录。

2. **聚类 (DBSCAN)**: 
    - 对去重后的数据应用 DBSCAN 聚类算法。`eps` 和 `min_samples` 参数控制聚类行为。聚类后的数据保存到 `--output_cluster` 指定的目录。

3. **聚类数量检查** (可选):
    - 如果 `--check_clusters` 设置为 `True`，脚本会打印每个 CSV 文件中（除噪声点外）的聚类数量。

4. **统计信息与绘图** (可选):
    - 如果 `--generate_stats` 设置为 `True`，脚本将生成统计摘要 CSV 文件，并绘制每个簇的 Z 轴最小值和最大值的二维图。文件保存在默认目录 `data/1/featureph3` 中。

5. **簇筛选**:
    - 脚本会删除满足条件（如 `z_max > 2500` 且点数小于 10）的簇，筛选后的数据保存到 `--output_filtered` 指定的目录。

### 输出示例:

- 去重后的文件: 保存在 `data/Deduplication1` 目录中。
- 聚类后的文件: 保存在 `data/cluster2` 目录中。
- 筛选后的文件: 保存在 `data/1/deletedata4` 目录中。
- 统计摘要与绘图 (如果 `--generate_stats=True`): 保存在 `data/1/featureph3` 目录中。

## 依赖库

- Python 3.x
- 所需 Python 库：
    - `pandas`
    - `numpy`
    - `scikit-learn`
    - `matplotlib`

你可以通过以下命令安装所需的库：

```bash
pip install -r requirements.txt
```

## 许可证


