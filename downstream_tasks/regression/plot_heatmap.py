import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_correlation_heatmap(
    input_file="top_30_features.xlsx",
    output_png="heatmap_SMA.png",
    output_svg="heatmap_SMA.svg",  # 添加SVG输出文件名
    figsize=(14, 12),
    threshold=0,
):
    """
    绘制相关系数热力图
    Args:
        input_file: 输入的Excel文件路径
        output_png: PNG输出文件路径
        output_svg: SVG输出文件路径
        figsize: 图片大小
        threshold: 显示的相关系数绝对值阈值
    """
    # 设置图表样式
    plt.style.use("seaborn")

    # 创建自定义红-灰-蓝颜色方案，添加更多颜色节点来控制渐变
    colors = [
        "#8b0000",  # 深红
        "#d73027",  # 中红
        "#f46d43",  # 浅红
        "#ff9999",  # 很浅的红
        "#ffffff",  # 白色（中心点）
        "#e0f3f8",  # 很浅的蓝
        "#abd9e9",  # 浅蓝
        "#74add1",  # 中蓝
        "#4575b4",  # 深蓝
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

    # 读取数据
    df = pd.read_excel(input_file)

    # 将第一列设置为索引（y轴标签）
    first_col = df.columns[0]
    df_plot = df.set_index(first_col)

    # 创建mask
    mask = np.abs(df_plot) < threshold

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    sns.heatmap(
        df_plot,
        annot=False,
        cmap=custom_cmap,
        center=0,
        square=True,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={
            "shrink": 0.8,
            "label": "Correlation Coefficient",
            "orientation": "horizontal",
        },
        mask=mask,
        ax=ax,
        vmin=-1,
        vmax=1,
    )

    # 设置标题和标签
    plt.title("SMA", pad=20, fontsize=16, fontweight="bold")
    plt.xlabel("Element Embeddings", fontsize=14, labelpad=10)
    plt.ylabel("Empirical Features", fontsize=14, labelpad=10)

    # 调整刻度标签
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # 添加网格线
    ax.set_axisbelow(True)

    # 调整布局
    plt.tight_layout()

    # 保存PNG图片
    plt.savefig(
        output_png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"PNG热力图已保存到: {output_png}")

    # 保存SVG图片
    plt.savefig(
        output_svg,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print(f"SVG热力图已保存到: {output_svg}")

    # 显示图片
    plt.show()


if __name__ == "__main__":
    try:
        plot_correlation_heatmap()
    except FileNotFoundError:
        print("错误：找不到输入文件")
    except Exception as e:
        print(f"发生错误: {e}")
