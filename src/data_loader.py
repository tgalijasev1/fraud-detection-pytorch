import pandas as pd
import torch
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

def prepare_data(zip_path='data/creditcard.zip', batch_size=256):
    """
    Opens the zip archive, extracts the CSV dataset, handles data preprocessing,
    and splits the dataset into Train/Val/Test subsets for unsupervised learning.
    """
    # 1. Open the zip file and dynamically find the correct CSV file path inside
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found inside the provided zip archive.")
        
        csv_name = csv_files[0]
        print(f"Extracting and loading: {csv_name}")
        
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
            
    # 2. Standardize column names to lowercase to avoid capitalization issues (e.g., cvv_Match vs cvv_match)
    df.columns = df.columns.str.lower()
    print("Available columns in dataset (lowercased):", list(df.columns))
    
    # 3. Feature Selection based on Exploratory Data Analysis (EDA)
    # Using lowercase matching to be 100% safe
    features_to_use = [
        'transactionamount', 'availablemoney', 'currentbalance', 'creditlimit',
        'cardpresent', 'cvv_match', 'isfraud'
    ]
    df = df[features_to_use].dropna()
    
    # Convert boolean indicators (True/False) to numeric binary flags (1/0)
    df['cardpresent'] = df['cardpresent'].astype(int)
    df['cvv_match'] = df['cvv_match'].astype(int)
    
    # 4. Standardize continuous numerical values (Zero mean and unit variance)
    scaler = StandardScaler()
    num_cols = ['transactionamount', 'availablemoney', 'currentbalance', 'creditlimit']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # 5. Separate normal transactions from fraudulent ones
    normal_tx = df[df['isfraud'] == 0]
    fraud_tx = df[df['isfraud'] == 1]
    
    # 6. Split normal transactions: 80% Train, 10% Validation, 10% Test
    train_normal, temp_normal = train_test_split(normal_tx, test_size=0.2, random_state=42)
    val_normal, test_normal = train_test_split(temp_normal, test_size=0.5, random_state=42)
    
    # Drop labels for unsupervised training (Autoencoder learns normal patterns only)
    X_train = train_normal.drop(['isfraud'], axis=1).values
    X_val = val_normal.drop(['isfraud'], axis=1).values
    
    # 7. Create the final evaluation test set containing both normal data and fraud samples
    test_set = pd.concat([test_normal, fraud_tx])
    X_test = test_set.drop(['isfraud'], axis=1).values
    y_test = test_set['isfraud'].values
    
    # 8. Convert NumPy arrays into PyTorch Tensors and wrap them in DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.FloatTensor(X_val))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train.shape[1]
    
    return train_loader, val_loader, X_test, y_test, input_dim
