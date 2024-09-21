import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df = df.drop(columns=['id', 'dataset'], errors='ignore')

    # Convert categorical columns to numeric
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    
    # Use one-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Handle NaN values
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)

    # Define features and target
    X = df.drop(columns=['num'])
    y = df['num']

    # Apply SMOTE for balancing the classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    print(f"Shapes of the datasets after SMOTE:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test




























