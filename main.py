import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 生成围绕0波动的数据
np.random.seed(42)  # 设置随机种子以保证结果可重现
data = pd.Series(np.random.normal(0, 1, 20))  # 生成20个标准正态分布的数据

# 计算扩展 Z 分数
expanding_mean = data.expanding().mean()
expanding_std = data.expanding().std()
expanding_z_score = (data - expanding_mean) / expanding_std

# 绘制 Z 分数
plt.figure(figsize=(10, 5))
plt.plot(expanding_z_score, marker='o', linestyle='-', color='blue', label='Expanding Z Score')
plt.title('Expanding Z Score (围绕0波动)')
plt.xlabel('时间点')
plt.ylabel('Z Score')
plt.axhline(0, color='red', linestyle='--', linewidth=0.8)  # 添加零线
plt.axhline(2, color='gray', linestyle=':', linewidth=0.8)  # 添加上边界
plt.axhline(-2, color='gray', linestyle=':', linewidth=0.8)  # 添加下边界
plt.legend()
plt.grid(True)
plt.show()

# 打印统计信息
print("\nZ分数统计信息：")
print(expanding_z_score.describe())