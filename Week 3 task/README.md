# Week 3 – Unsupervised Learning and Clustering Analysis

## Objective

This project applies unsupervised learning and clustering techniques to the publicly available UCI Wine Dataset. The goal is to identify meaningful groups within the wine samples and analyze the characteristics of each cluster.

## Dataset

**Dataset:** UCI Wine Dataset
**Samples:** 178
**Features:** 13 chemical measurements
**Clustering Features:** Alcohol, Malic Acid, Ash, Alcalinity of Ash, Magnesium, Total Phenols, Flavanoids, Nonflavanoid Phenols, Proanthocyanins, Color Intensity, Hue, OD280/OD315 of Diluted Wines, Proline.

The dataset was standardized before clustering because the variables have different scales.

## Methodology

The analysis includes:

1. Data preprocessing and feature standardization
2. Exploratory analysis
3. K-Means clustering
4. Elbow method for selecting the number of clusters
5. Silhouette analysis
6. PCA-based 2D visualization
7. Cluster profile analysis
8. Hierarchical clustering as a secondary comparison

## Clustering Results

The analysis identified **3 clusters** as the most appropriate solution.

The K-Means model with `k = 3` achieved a silhouette score of approximately **0.285**.

PCA was used to visualize the clusters in two dimensions. The first two principal components explain approximately **55.4% of the total variance**.

The cluster profiles show that the groups differ primarily in chemical characteristics such as alcohol, flavanoids, total phenols, color intensity, and proline.

## Interpretation

### Cluster 0

Represents wines with a distinct chemical profile characterized by relatively strong values for several phenolic and compositional measurements.

### Cluster 1

Contains samples with a contrasting chemical profile and is separated from the other groups across important standardized features.

### Cluster 2

Forms a third group with its own combination of chemical characteristics, particularly in features related to phenolic composition and intensity.

These clusters can potentially support wine segmentation, quality analysis, product differentiation, and exploratory research.

## Visualizations

The project contains:

* Elbow curve
* Silhouette score comparison
* PCA cluster visualization
* Cluster profile heatmap
* Hierarchical clustering dendrogram
* Silhouette plot

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SciPy
* Jupyter Notebook

## Files

The repository contains the complete report, figures, analysis outputs, and reproducible clustering workflow.

## Conclusion

Unsupervised clustering successfully identified three meaningful groups within the Wine Dataset. K-Means provided an effective and interpretable segmentation, while hierarchical clustering and PCA visualization supported the overall structure discovered in the data.


