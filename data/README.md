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

### 3. Housing data (`ames_housing.csv`, actually Boston Housing)

-   **File:** `ames_housing.csv`
-   **What it actually is:** the classic **Boston Housing** dataset (Harrison & Rubinfeld, 1978), not the Ames, Iowa dataset the file name suggests. It has **506 rows and 14 numeric columns** (13 features plus the median home value, `MEDV`, last), and **no header row**: `pd.read_csv` with its defaults turns the first record into the column names. Read it with `header=None`.
-   **Task:** Predict the median value of owner-occupied homes (the last column).
-   **Properties:**
    -   Records: 506
    -   Features: 13 numeric + 1 target (the last column)
-   **Caveats:**
    -   The earlier version of this README described the Ames dataset (2,919 records, 79 features, many categorical). That was wrong for this file, and the tests that load it only see anonymous numeric columns.
    -   One Boston feature (`B`) is derived from the racial composition of each town, and the dataset has been withdrawn from some libraries (scikit-learn removed `load_boston`) for that reason. Consider replacing it with the real Ames data before using it for anything beyond a numeric smoke test.
