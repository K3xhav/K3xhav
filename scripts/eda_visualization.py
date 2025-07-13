import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Load and prepare the dataset
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df

# Basic statistics
def display_basic_stats(df):
    print("Dataset Info:")
    print(df.info())
    print("\nDataset Shape:", df.shape)
    print("\nBasic Statistics:")
    print(df.describe())
    print("\nSpecies Distribution:")
    print(df['species_name'].value_counts())

# Create visualizations
def create_visualizations(df):
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Iris Dataset Exploratory Data Analysis', fontsize=16)
    
    # 1. Distribution of features
    df.iloc[:, :4].hist(bins=20, ax=axes[0, 0])
    axes[0, 0].set_title('Feature Distributions')
    
    # 2. Correlation heatmap
    correlation_matrix = df.iloc[:, :4].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[0, 1])
    axes[0, 1].set_title('Feature Correlation Heatmap')
    
    # 3. Pairplot for species
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', 
                    hue='species_name', ax=axes[1, 0])
    axes[1, 0].set_title('Sepal Length vs Petal Length')
    
    # 4. Box plot for species comparison
    df_melted = df.melt(id_vars=['species_name'], 
                        value_vars=['sepal length (cm)', 'sepal width (cm)', 
                                   'petal length (cm)', 'petal width (cm)'])
    sns.boxplot(data=df_melted, x='variable', y='value', hue='species_name', ax=axes[1, 1])
    axes[1, 1].set_title('Feature Comparison by Species')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/iris_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df = load_iris_data()
    display_basic_stats(df)
    create_visualizations(df)
    print("\nAnalysis complete! Check outputs/iris_analysis.png for visualizations.")
