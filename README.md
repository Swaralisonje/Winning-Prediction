# Olympics Medal Predictor

This project is a web application that predicts the likelihood of a country winning medals in various Olympic sports. The app leverages machine learning models to analyze historical Olympic data and provides predictions and visual insights for users. Additionally, embedded Power BI reports offer interactive data exploration.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Olympics Medal Predictor analyzes Olympic participation and medal-winning patterns for different countries across various sports. Users select a sport, and the app:
- Displays key statistics on top-performing countries.
- Predicts the chances of future medal wins using Naive Bayes and Logistic Regression.
- Presents model performance metrics for comparison.

Embedded Power BI reports provide additional visualizations and insights into the data.

## Features

- **Interactive Sport Selection**: Users can input a specific sport to view relevant data and predictions.
- **Top 10 Countries Statistics**: Ranks countries by win percentage and displays participation, medal count, and predicted winning chances.
- **Machine Learning Predictions**: Uses Naive Bayes and Logistic Regression models to predict medal-winning chances.
- **Model Performance Comparison**: Shows accuracy, precision, recall, and F1 scores for each model.
- **Visualization**: Histogram visualization of win percentages by country, with additional Power BI reports embedded for in-depth analysis.

## Technologies Used

- **Python**: Flask, Pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn
- **Power BI**: Embedded reports for extended data analysis and visualization
- **HTML/CSS**: For frontend user interface

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/olympics-medal-predictor.git
   cd olympics-medal-predictor
