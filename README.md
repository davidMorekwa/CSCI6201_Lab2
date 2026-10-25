# Predicting California House Values

CSCI 6201 Final Project, Lab 2 Checkpoint

## Team Members

- David Morekwa
- Yehowashidaa Ashong-Kotei

## Dataset

**California Housing** (from scikit-learn)

- 20,640 samples (each one is a district in California, from the 1990 US Census)
- 8 features: median income, house age, average rooms, average bedrooms, population, average occupancy, latitude and longitude
- Target: `MedHouseVal`, the median house value in the district (in units of $100,000)
- No missing values

Note: this dataset is listed under scikit-learn's Real World Datasets rather than the Toy Datasets.

## Problem Type

**Regression**: we are predicting a number (house value), not a category.

## Research Questions

1. Can the available features (income, house age, location, etc.) be used to predict the median house value of a district?
2. Which features are most strongly related to house value?
3. How well does a simple Linear Regression model perform on this data?

## Project Structure

| File | What it does |
|------|--------------|
| `load.py` | Loads the California Housing dataset from scikit-learn |
| `explore.py` | Exploratory analysis: shape, statistics, missing values, correlations |
| `visualize.py` | Creates a geographic scatter plot and a histogram of house values |
| `model.py` | Splits the data and trains a Linear Regression model |
| `main.py` | Runs the full workflow: load, visualize, train and evaluate |
| `test.py` | Unit tests for the helper functions and dataset loading |
| `figures/` | Saved charts |

## How to Run

```
pip install -r requirements.txt
python explore.py     # exploratory analysis
python main.py        # charts + model
python test.py        # run tests
```

## Current Progress

- Dataset loads successfully, with 20,640 samples and 8 features and no missing values.
- Exploratory analysis is complete: descriptive statistics, missing-value checks and correlations with the target.
- Key findings from exploration:
  - The average median house value is about $207,000.
  - Median income has by far the strongest link to house value (correlation 0.69). No other feature is above 0.15.
  - 965 districts are capped at exactly $500,000, so the true values for those districts may be higher.
  - `AveRooms` has some extreme outliers (up to about 142 rooms per household), likely from districts with very few households.
- Two visualizations created:
  - **Geographic scatter plot** (`figures/geo-scatter.png`): The most expensive districts (yellow) are concentrated along the coast, especially around the San Francisco Bay Area and Los Angeles, while inland and northern districts are mostly lower in value. This suggests location matters a lot, even though latitude and longitude alone showed weak straight-line correlations with house value.
  - **Histogram of house values** (`figures/house-value-histogram.png`): Most districts have median house values between about $100,000 and $250,000, and the distribution is skewed to the right. There is a sharp spike at $500,000, which comes from the cap in the original data rather than real prices.
- A Linear Regression model was trained on 70% of the data and tested on the other 30%.
- **Preliminary result:** Mean Squared Error = **0.53**. Taking the square root (about 0.73) puts this back in the original units, so the model's predictions are typically off by roughly $73,000. Since the average house value is about $207,000, this is a reasonable start but leaves clear room for improvement.

## Next Steps for Final Project

**1. What is working so far?**
The full workflow runs: loading the data, exploring it, creating charts, training a Linear Regression model and evaluating it. Basic unit tests pass.

**2. What part of the project still needs work?**
We only report one evaluation metric (MSE) and use a single model. We have not yet dealt with the house values that are capped at $500,000.

**3. What do we plan to add or improve?**
- Add more metrics (Mean Absolute Error and R-squared) so results are easier to interpret
- Try other models, such as a Decision Tree or Random Forest, and compare them with Linear Regression
- Look at which features matter most to the model
- Investigate how the $500,000 cap affects predictions

**4. Questions for instructor feedback**
- Should we remove the capped house values before modeling, or keep them?
