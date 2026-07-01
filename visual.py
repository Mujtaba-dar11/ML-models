import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Dataset Banana
data = {
    'Study Hours': [2, 3, 4, 5, 6, 7, 8, 9],
    'Test Scores': [50, 60, 65, 70, 75, 80, 85, 95]
}
df = pd.DataFrame(data)

# ==========================================
# GRAPH 1: Scatter Plot (Dots wala graph)
# ==========================================
plt.figure(figsize=(6, 4)) # Graph ka size set karna
plt.title("Scatter Plot: Study Hours vs Test Scores")

# Seaborn ka scatterplot
sns.scatterplot(x=df['Study Hours'], y=df['Test Scores'], color='blue', s=100)

plt.show()


# ==========================================
# GRAPH 2: Seaborn Heatmap (Correlation Matrix)
# ==========================================
# Pehle correlation nikalna laazmi hai
matrix = df.corr()

plt.figure(figsize=(6, 4))
plt.title("Heatmap: Correlation Matrix")

# Seaborn ka heatmap (annot=True se numbers aayenge)
sns.heatmap(matrix, annot=True, cmap='coolwarm')

plt.show()