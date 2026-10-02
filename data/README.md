# Datasets for Row2Vec

This directory contains a collection of classic, public-domain datasets used for demonstrating and testing the `Row2Vec` library. Each dataset features a mix of numerical and categorical data, making them ideal for representation learning tasks.

---

### 1. Adult Income Dataset

-   **File:** `adult.csv`
-   **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
-   **Task:** Predict whether an individual's annual income exceeds $50,000.
-   **Description:** This dataset was extracted from the 1994 U.S. Census database. It's a very popular dataset for binary classification tasks and is a prime example of a typical tabular dataset with mixed data types.
-   **Properties:**
    -   Records: 48,842
    -   Features: 14 + 1 target (`income`)
-   **Features:**
    -   `age`: continuous.
    -   `workclass`: categorical.
    -   `fnlwgt`: continuous.
    -   `education`: categorical.
    -   `education-num`: continuous.
    -   `marital-status`: categorical.
    -   `occupation`: categorical.
    -   `relationship`: categorical.
    -   `race`: categorical.
    -   `sex`: categorical.
    -   `capital-gain`: continuous.
    -   `capital-loss`: continuous.
    -   `hours-per-week`: continuous.
    -   `native-country`: categorical.
    -   `income`: categorical target (`<=50K` or `>50K`).

---

### 2. Titanic Dataset

-   **File:** `titanic.csv`
-   **Source:** [Kaggle / Stanford](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv)
-   **Task:** Predict passenger survival on the Titanic.
-   **Description:** This is arguably the most famous dataset in data science, often used as a "hello world" for predictive modeling. It contains demographic and travel information for passengers aboard the RMS Titanic.
-   **Properties:**
    -   Records: 887
    -   Features: 7 + 1 target (`Survived`)
-   **Features:**
    -   `Survived`: categorical target (0 = No, 1 = Yes).
    -   `Pclass`: categorical (1st, 2nd, 3rd class).
    -   `Name`: string (usually excluded or engineered).
    -   `Sex`: categorical.
    -   `Age`: continuous.
    -   `Siblings/Spouses Aboard`: continuous.
    -   `Parents/Children Aboard`: continuous.
    -   `Fare`: continuous.

---

### 3. Ames Housing Dataset

-   **File:** `ames_housing.csv`
-   **Source:** [Dean De Cock, "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project", *Journal of Statistics Education* 19(3), 2011](https://jse.amstat.org/v19n3/decock.pdf). This copy is the Kaggle "House Prices" training set derived from it, as published on [OpenML (id 42165)](https://www.openml.org/d/42165).
-   **Task:** Predict the final sale price of homes in Ames, Iowa (`SalePrice`, numeric).
-   **Description:** A detailed dataset of residential sales with a large number of features, many of them categorical, and missing values in 19 columns. It is a good test of mixed-type preprocessing at a width where most columns are not numeric.
-   **Properties:**
    -   Records: 1,460
    -   Columns: `Id`, 79 features (43 categorical, 36 numeric) and the target `SalePrice`
-   **Notes:**
    -   `Id` is a row counter, not a feature; drop it before embedding.
    -   In several columns a missing value means the feature is absent (no pool, no alley, no garage); `PoolQC` is 99% missing for that reason. `pandas.read_csv` reads the literal `NA` as missing too, which matches that convention.
    -   This is the 1,460-row training half of the Kaggle split, not the full 2,930-row Ames table.
    -   An earlier version of this repository shipped the Boston Housing data under this file name. It was replaced (and the Boston data removed, partly because one of its features is derived from the racial composition of each town).
