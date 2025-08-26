#!/usr/bin/env python3
"""
Row2Vec Missing Value Imputation - Quick Start Guide

This script demonstrates how to use Row2Vec's intelligent missing value
imputation system with real-world scenarios.
"""

import pandas as pd
import numpy as np
from row2vec import ImputationConfig, AdaptiveImputer, MissingPatternAnalyzer

def load_sample_data():
    """Create a realistic dataset with missing values."""
    np.random.seed(42)
    
    # Simulate a customer dataset
    n_customers = 500
    data = {
        'age': np.random.normal(35, 12, n_customers),
        'income': np.random.lognormal(10.5, 0.5, n_customers),
        'credit_score': np.random.normal(650, 100, n_customers),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_customers, p=[0.4, 0.3, 0.2, 0.1]),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers),
        'has_car': np.random.choice([True, False], n_customers, p=[0.7, 0.3])
    }
    
    df = pd.DataFrame(data)
    
    # Introduce realistic missing patterns
    # 1. Age missing for younger people (privacy concerns)
    young_mask = (df['age'] < 25) & (np.random.random(n_customers) < 0.3)
    df.loc[young_mask, 'age'] = np.nan
    
    # 2. Income missing for high earners (privacy) and unemployed
    high_income_mask = (df['income'] > df['income'].quantile(0.9)) & (np.random.random(n_customers) < 0.4)
    low_income_mask = (df['income'] < df['income'].quantile(0.1)) & (np.random.random(n_customers) < 0.2)
    df.loc[high_income_mask | low_income_mask, 'income'] = np.nan
    
    # 3. Credit score missing randomly (some people haven't been scored)
    credit_missing = np.random.random(n_customers) < 0.15
    df.loc[credit_missing, 'credit_score'] = np.nan
    
    # 4. Education missing more often for older generations
    old_mask = (df['age'] > 60) & (np.random.random(n_customers) < 0.25)
    df.loc[old_mask, 'education'] = np.nan
    
    return df

def demonstrate_basic_usage():
    """Demonstrate basic imputation with default settings."""
    print("=" * 60)
    print("1. BASIC USAGE - Default Adaptive Imputation")
    print("=" * 60)
    
    # Load data
    df = load_sample_data()
    print(f"Original data shape: {df.shape}")
    
    # Show missing data summary
    print("\nMissing values per column:")
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")
    
    # Apply default imputation
    imputer = AdaptiveImputer(ImputationConfig())
    df_imputed = imputer.fit_transform(df)
    
    print(f"\nAfter imputation:")
    print(f"  Remaining missing values: {df_imputed.isna().sum().sum()}")
    print(f"  Data shape preserved: {df_imputed.shape == df.shape}")
    
    return df, df_imputed

def demonstrate_pattern_analysis():
    """Demonstrate missing pattern analysis."""
    print("\n" + "=" * 60)
    print("2. PATTERN ANALYSIS - Understanding Missing Data")
    print("=" * 60)
    
    df = load_sample_data()
    
    # Analyze patterns
    config = ImputationConfig()
    analyzer = MissingPatternAnalyzer(config)
    analysis = analyzer.analyze(df)
    
    print(f"Missing Data Analysis:")
    print(f"  Total missing values: {analysis['total_missing']}")
    print(f"  Overall missing rate: {analysis['missing_percentage']:.1f}%")
    print(f"  Columns affected: {analysis['columns_with_missing']}")
    
    print(f"\nRecommendations by column:")
    for col, rec in analysis['recommendations'].items():
        if rec:  # Only show columns with missing data
            print(f"  {col}:")
            print(f"    Missing: {rec['missing_percentage']:.1f}%")
            print(f"    Recommended: {rec['suggested_strategy']}")
            print(f"    Reason: {rec['reasoning']}")

def demonstrate_custom_strategies():
    """Demonstrate custom imputation strategies."""
    print("\n" + "=" * 60)
    print("3. CUSTOM STRATEGIES - Tailored Imputation")
    print("=" * 60)
    
    df = load_sample_data()
    
    # Strategy 1: Fast and simple (for quick prototyping)
    print("\nStrategy 1: Fast and Simple")
    config_fast = ImputationConfig(
        numeric_strategy='mean',        # Fastest for numeric
        categorical_strategy='mode',    # Fastest for categorical
        prefer_speed=True
    )
    
    imputer_fast = AdaptiveImputer(config_fast)
    df_fast = imputer_fast.fit_transform(df)
    print(f"  Completed in fast mode: {df_fast.isna().sum().sum()} missing values remaining")
    
    # Strategy 2: High accuracy (for production)
    print("\nStrategy 2: High Accuracy")
    config_accurate = ImputationConfig(
        numeric_strategy='knn',         # More accurate for numeric
        categorical_strategy='mode',    # Best available for categorical
        prefer_speed=False,
        knn_neighbors=10,              # More neighbors for stability
        preserve_missing_patterns=True # Keep track of what was missing
    )
    
    imputer_accurate = AdaptiveImputer(config_accurate)
    df_accurate = imputer_accurate.fit_transform(df)
    
    # Check for missing indicators
    indicator_cols = [col for col in df_accurate.columns if col.endswith('_was_missing')]
    print(f"  Completed in accurate mode: {df_accurate.isna().sum().sum()} missing values remaining")
    print(f"  Missing pattern indicators created: {len(indicator_cols)}")
    
    # Strategy 3: Domain-specific (custom for this dataset)
    print("\nStrategy 3: Domain-Specific")
    config_custom = ImputationConfig(
        numeric_strategy='median',      # Robust to income outliers
        categorical_strategy='mode',
        preserve_missing_patterns=True,
        missing_threshold=0.5          # More tolerant of missing data
    )
    
    imputer_custom = AdaptiveImputer(config_custom)
    df_custom = imputer_custom.fit_transform(df)
    print(f"  Completed with custom strategy: {df_custom.isna().sum().sum()} missing values remaining")

def demonstrate_integration():
    """Demonstrate integration with machine learning pipelines."""
    print("\n" + "=" * 60)
    print("4. INTEGRATION - ML Pipeline Integration")
    print("=" * 60)
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        
        # Create dataset with target variable
        df = load_sample_data()
        
        # Create a synthetic target (e.g., loan approval)
        np.random.seed(42)
        # Higher approval probability for higher income and credit score
        income_norm = (df['income'].fillna(df['income'].median()) - df['income'].fillna(df['income'].median()).mean()) / df['income'].fillna(df['income'].median()).std()
        credit_norm = (df['credit_score'].fillna(df['credit_score'].median()) - df['credit_score'].fillna(df['credit_score'].median()).mean()) / df['credit_score'].fillna(df['credit_score'].median()).std()
        approval_prob = 1 / (1 + np.exp(-(income_norm + credit_norm)))  # Sigmoid
        df['loan_approved'] = np.random.binomial(1, approval_prob)
        
        # Split features and target
        X = df.drop('loan_approved', axis=1)
        y = df['loan_approved']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Missing values in training set: {X_train.isna().sum().sum()}")
        
        # Create and fit imputer on training data
        imputer = AdaptiveImputer(ImputationConfig(
            numeric_strategy='knn',
            categorical_strategy='mode',
            preserve_missing_patterns=False  # Don't need indicators for this example
        ))
        
        # Fit on training data and transform both sets
        X_train_imputed = imputer.fit(X_train).transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        print(f"After imputation:")
        print(f"  Training missing values: {X_train_imputed.isna().sum().sum()}")
        print(f"  Test missing values: {X_test_imputed.isna().sum().sum()}")
        
        # Convert categorical columns to numeric for sklearn
        for col in X_train_imputed.select_dtypes(include=['object', 'bool']).columns:
            # Simple label encoding for demonstration
            unique_values = list(set(X_train_imputed[col].unique()) | set(X_test_imputed[col].unique()))
            value_map = {val: idx for idx, val in enumerate(unique_values)}
            X_train_imputed[col] = X_train_imputed[col].map(value_map)
            X_test_imputed[col] = X_test_imputed[col].map(value_map)
        
        # Train a simple model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()
        
        print(f"\nML Pipeline Results:")
        print(f"  Model accuracy: {accuracy:.3f}")
        print(f"  Successfully trained on imputed data!")
        
    except ImportError:
        print("sklearn not available - skipping ML integration demo")

def main():
    """Run all demonstrations."""
    print("Row2Vec Missing Value Imputation - Quick Start Guide")
    print("=" * 60)
    print()
    print("This guide shows how to use Row2Vec's intelligent imputation system")
    print("for handling missing values in your datasets.")
    print()
    
    # Run demonstrations
    demonstrate_basic_usage()
    demonstrate_pattern_analysis()
    demonstrate_custom_strategies()
    demonstrate_integration()
    
    print("\n" + "=" * 60)
    print("QUICK START COMPLETE! 🎉")
    print("=" * 60)
    print()
    print("Key takeaways:")
    print("1. Default settings work well for most datasets")
    print("2. Pattern analysis helps understand your missing data")
    print("3. Custom strategies give you full control")
    print("4. Seamless integration with ML pipelines")
    print()
    print("Next steps:")
    print("- Try with your own dataset")
    print("- Experiment with different strategies")
    print("- Use pattern analysis to guide your choices")
    print("- Integrate with your existing ML workflows")

if __name__ == "__main__":
    main()
