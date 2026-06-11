import torch
from src.data_loader import prepare_data
from src.model import AnomalyAutoencoder
from src.training import train_model
from src.evaluation import evaluate_model

def main():
    # Set device to GPU if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load and preprocess the dataset from the zip file
    print("\n--- Step 1: Preparing Data ---")
    train_loader, val_loader, X_test, y_test, input_dim = prepare_data('data/creditcard.zip', batch_size=256)
    print(f"Data ready. Input dimensions (number of features): {input_dim}")
    
    # 2. Initialize the Autoencoder model using the dynamic input dimension
    print("\n--- Step 2: Initializing Model ---")
    model = AnomalyAutoencoder(input_dim=input_dim)
    print(model)
    
    # 3. Train the model exclusively on legitimate transactions
    print("\n--- Step 3: Training Model ---")
    # Setting epochs to 20 for a stable training cycle
    trained_model, history = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=20, 
        lr=0.001, 
        device=device
    )
    
    # 4. Save the trained model weights for future inference or grading
    print("\n--- Step 4: Saving Trained Model ---")
    # PyTorch looks for a directory, make sure you or the script creates it locally
    import os
    os.makedirs('models', exist_ok=True)
    torch.save(trained_model.state_dict(), 'models/autoencoder_fraud.pth')
    print("Model parameters successfully saved to 'models/autoencoder_fraud.pth'")
    
    # 5. Evaluate the model performance on the mixed test set
    print("\n--- Step 5: Evaluating Model ---")
    evaluation_results = evaluate_model(
        model=trained_model, 
        X_test=X_test, 
        y_test=y_test, 
        device=device
    )
    
    print("\nPipeline execution completed successfully!")

if __name__ == '__main__':
    main()
