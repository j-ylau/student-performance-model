# Educational Performance Predictor

Welcome to the Educational Performance Predictor repository! This project showcases my skills in data preprocessing, machine learning model training, and software design. The project aims to predict educational performance using real-world data.

## Table of Contents
- [Introduction](#introduction)
- [Code](#code)
- [Tools](#tools)
- [Design Decisions](#design-decisions)

## Introduction

The **Educational Performance Predictor** is a machine learning project that predicts educational performance based on various features such as attendance, enrollment, demographics, and more. The project uses a Random Forest Regressor model to make predictions. The main steps of the project include data preprocessing, model training, and prediction.

## Code

The project is organized into the following structure:

```
Educational_Performance_Predictor/
├── data/
│   └── education_data.csv
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── predict.py
├── notebooks/
│   └── Data_Exploration.ipynb
└── requirements.txt
```

The `data` folder contains the raw CSV data, while the `src` folder contains the Python scripts responsible for data preprocessing, model training, and prediction. The `notebooks` folder contains Jupyter Notebooks used for data exploration.

## Tools

The project utilizes the following tools and libraries:

- **Python**: The core programming language used for development.
- **Pandas**: Data manipulation and analysis library.
- **NumPy**: Fundamental package for scientific computing with Python.
- **scikit-learn**: Machine learning library for various algorithms and tools.
- **matplotlib**: Plotting library for data visualization.
- **Jupyter Notebook**: Interactive environment for data analysis.

## Design Decisions

Several important design decisions were made to create an impressive Educational Performance Predictor:

1. **Data Preprocessing**: To handle data inconsistencies, a custom function `convert_percent_to_float` was implemented to convert percentage strings to float numbers. This preprocessing step ensures that the data is ready for modeling.

2. **Feature Selection**: Features were selected based on domain knowledge and relevance to educational performance. This selection process was crucial to creating a streamlined model that focuses on the most impactful variables.

3. **Model Selection**: The Random Forest Regressor was chosen due to its ability to handle complex relationships in data. This choice was backed by experimentation and comparison with other models.

4. **Scalability**: The project is designed to accommodate larger datasets by using libraries like Pandas and scikit-learn, which are known for their scalability and efficient data handling.

5. **Code Organization**: The project's modular organization allows for easy maintenance, extension, and testing. The separation of preprocessing, model training, and prediction into different modules enhances code readability and reusability.

By exploring the Educational Performance Predictor repository, recruiters can gain insights into my coding skills, software design decisions, and ability to create end-to-end machine learning projects. Feel free to explore the code, tools used, and design choices to see my proficiency in action.