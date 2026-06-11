import os
import torch
import yaml
from src.data_loader import prepare_data
from src.model import AnomalyAutoencoder
from src.training import train_model
from src.evaluation import evaluate_model

def load_config(config_path="config.yaml"):
    """Loads configuration parameters from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load settings from the configuration file
    config = load_config()
    
    # Set device to GPU if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load and preprocess the dataset using batch size from config
    print("\n--- Step 1: Preparing Data ---")
    train_loader, val_loader, X_test, y_test, input_dim = prepare_data(
        zip_path=config['data']['zip_path'], 
        batch_size=config['data']['batch_size']
    )
    print(f"Data ready. Input dimensions (number of features): {input_dim}")
    
    # 2. Initialize the Autoencoder model using the dynamic input dimension
    print("\n--- Step 2: Initializing Model ---")
    model = AnomalyAutoencoder(input_dim=input_dim)
    print(model)
    
    # 3. Train the model using hyperparameters defined in config.yaml
    print("\n--- Step 3: Training Model ---")
    trained_model, history = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=config['training']['epochs'], 
        lr=config['training']['learning_rate'], 
        device=device
    )
    
    # 4. Save the trained model weights for future inference or grading
    print("\n--- Step 4: Saving Trained Model ---")
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
    
    print("\nPipeline execution completed successfully using config.yaml parameters!")

if __name__ == '__main__':
    main()
