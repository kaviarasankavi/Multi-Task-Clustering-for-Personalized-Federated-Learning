import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("MCFL Algorithm Implementation on Electricity Dataset")
print("=" * 80)

# ============================================================================
# Step 1: Load and Preprocess Electricity Dataset
# ============================================================================
print("\n[Step 1] Loading and preprocessing Electricity Dataset...")

# Load the dataset
df = pd.read_csv('Electricity Dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Convert Month to datetime
df['Month'] = pd.to_datetime(df['Month'])

# Select numerical features (exclude Month column)
feature_cols = df.columns[1:].tolist()
print(f"\nFeatures used: {len(feature_cols)} columns")

# Sort by date
df = df.sort_values('Month').reset_index(drop=True)

# Prepare data for time series forecasting
# We'll use a sliding window approach: use past 'window_size' months to predict next month
window_size = 6
target_col = 'Electricity Net Generation, Total'  # Main target variable

def create_sequences(data, window_size):
    """Create sequences for time series forecasting"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(df[feature_cols].values)

# Create sequences
X, y = create_sequences(data_normalized, window_size)
print(f"\nSequence data shape: X={X.shape}, y={y.shape}")

# Split into train and test
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ============================================================================
# Step 2: Split Data into Multiple Clients (Non-IID Distribution)
# ============================================================================
print("\n[Step 2] Splitting data into multiple clients...")

num_clients = 10
print(f"Number of clients: {num_clients}")

# Create non-IID distribution by splitting data based on time periods
# Each client gets data from different time periods (simulating different regions/patterns)
def split_data_non_iid(X, y, num_clients):
    """Split data into non-IID partitions for clients"""
    client_data = []
    n_samples = len(X)

    # Split data into overlapping chunks for non-IID distribution
    samples_per_client = n_samples // num_clients

    for i in range(num_clients):
        # Each client gets a different portion with some overlap
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client + samples_per_client // 2
        end_idx = min(end_idx, n_samples)

        # Add some randomness
        indices = list(range(start_idx, end_idx))
        np.random.shuffle(indices)

        client_X = X[indices]
        client_y = y[indices]

        client_data.append({
            'X': torch.FloatTensor(client_X),
            'y': torch.FloatTensor(client_y),
            'n_samples': len(client_X)
        })

        print(f"Client {i+1}: {len(client_X)} samples")

    return client_data

client_data = split_data_non_iid(X_train, y_train, num_clients)

# Prepare test data
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# ============================================================================
# Step 3: Define Neural Network Model (Encoder + Predictor)
# ============================================================================
print("\n[Step 3] Defining neural network model...")

class MCFLModel(nn.Module):
    """
    MCFL Model with separate Encoder and Predictor components
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MCFLModel, self).__init__()

        # Encoder part (θe) - shared across clusters
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )

        # Predictor part (θp) - cluster-specific
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )

    def forward(self, x):
        # Flatten the input (batch_size, window_size, features)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Encode
        encoded = self.encoder(x)

        # Predict
        output = self.predictor(encoded)

        return output

    def get_encoder_params(self):
        """Get encoder parameters"""
        return {name: param.clone() for name, param in self.encoder.named_parameters()}

    def get_predictor_params(self):
        """Get predictor parameters"""
        return {name: param.clone() for name, param in self.predictor.named_parameters()}

    def set_encoder_params(self, params):
        """Set encoder parameters"""
        for name, param in self.encoder.named_parameters():
            param.data = params[name].data.clone()

    def set_predictor_params(self, params):
        """Set predictor parameters"""
        for name, param in self.predictor.named_parameters():
            param.data = params[name].data.clone()

# Model hyperparameters
input_size = window_size * len(feature_cols)
hidden_size = 128
output_size = len(feature_cols)

print(f"Model architecture: Input={input_size}, Hidden={hidden_size}, Output={output_size}")

# ============================================================================
# Step 4: Local Client Training Function
# ============================================================================
print("\n[Step 4] Implementing local client training...")

def train_local_model(model, client_data, epochs, lr):
    """
    Train model locally on client data
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(client_data['X'])
        loss = criterion(outputs, client_data['y'])

        # Backward pass
        loss.backward()
        optimizer.step()

    return model, loss.item()

# ============================================================================
# Step 5: Server-Side K-means Clustering
# ============================================================================
print("\n[Step 5] Implementing K-means clustering...")

def cluster_clients(client_models, num_clusters):
    """
    Apply K-means clustering on client model parameters
    """
    # Extract model parameters and flatten them
    param_vectors = []

    for model in client_models:
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        param_vector = torch.cat(params).cpu().numpy()
        param_vectors.append(param_vector)

    param_vectors = np.array(param_vectors)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(param_vectors)

    # Group clients by cluster
    clusters = {i: [] for i in range(num_clusters)}
    for client_idx, cluster_id in enumerate(cluster_labels):
        clusters[cluster_id].append(client_idx)

    return clusters, cluster_labels

# ============================================================================
# Step 6: Cluster Aggregation Function
# ============================================================================
print("\n[Step 6] Implementing cluster aggregation...")

def aggregate_cluster_models(client_models, cluster_clients, client_data):
    """
    Aggregate models within each cluster (weighted by number of samples)
    """
    aggregated_model = MCFLModel(input_size, hidden_size, output_size)

    # Calculate total samples in cluster
    total_samples = sum([client_data[i]['n_samples'] for i in cluster_clients])

    # Initialize aggregated parameters
    aggregated_params = {}
    for name, param in client_models[cluster_clients[0]].named_parameters():
        aggregated_params[name] = torch.zeros_like(param.data)

    # Weighted aggregation
    for client_idx in cluster_clients:
        weight = client_data[client_idx]['n_samples'] / total_samples
        for name, param in client_models[client_idx].named_parameters():
            aggregated_params[name] += weight * param.data

    # Set aggregated parameters
    for name, param in aggregated_model.named_parameters():
        param.data = aggregated_params[name]

    return aggregated_model

# ============================================================================
# Step 7: Client Adaptation with Weighted Ensemble
# ============================================================================
print("\n[Step 7] Implementing client adaptation...")

def adapt_client_model(client_idx, cluster_id, cluster_encoders, cluster_predictor,
                       client_data_item, adaptation_epochs, lr):
    """
    Adapt client model using weighted ensemble of cluster encoders
    """
    # Create client-specific model
    client_model = MCFLModel(input_size, hidden_size, output_size)

    # Initialize weights for encoder ensemble (learnable)
    num_clusters = len(cluster_encoders)
    ensemble_weights = nn.Parameter(torch.ones(num_clusters) / num_clusters)

    # Set predictor from client's cluster
    client_model.set_predictor_params(cluster_predictor)

    # Simple adaptation: use encoder from client's own cluster
    # In a full implementation, we would learn weights for ensemble
    client_model.set_encoder_params(cluster_encoders[cluster_id])

    # Fine-tune on local data
    client_model, loss = train_local_model(client_model, client_data_item,
                                          adaptation_epochs, lr)

    return client_model

# ============================================================================
# Step 8: Main MCFL Training Loop
# ============================================================================
print("\n[Step 8] Implementing main MCFL training loop...")

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test data"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = nn.MSELoss()(predictions, y_test).item()
        mae = torch.mean(torch.abs(predictions - y_test)).item()
    return mse, mae

def run_mcfl(num_rounds, num_clusters, local_epochs, adaptation_epochs, lr,
             client_selection_ratio=0.5):
    """
    Main MCFL Algorithm
    """
    print("\n" + "=" * 80)
    print("Starting MCFL Training")
    print("=" * 80)
    print(f"Rounds: {num_rounds}")
    print(f"Clusters: {num_clusters}")
    print(f"Local epochs: {local_epochs}")
    print(f"Adaptation epochs: {adaptation_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Client selection ratio: {client_selection_ratio}")

    # Initialize global model
    global_model = MCFLModel(input_size, hidden_size, output_size)

    # Training history
    history = {
        'train_loss': [],
        'test_mse': [],
        'test_mae': [],
        'cluster_assignments': []
    }

    for round_idx in range(num_rounds):
        print(f"\n{'='*80}")
        print(f"Round {round_idx + 1}/{num_rounds}")
        print(f"{'='*80}")

        # Step 1: Select subset of clients
        num_selected = max(1, int(num_clients * client_selection_ratio))
        selected_clients = np.random.choice(num_clients, num_selected, replace=False)
        print(f"Selected clients: {selected_clients.tolist()}")

        # Step 2: Local Training
        print("\n[Local Training]")
        client_models = []
        train_losses = []

        for i, client_idx in enumerate(selected_clients):
            # Initialize client model with global model
            client_model = deepcopy(global_model)

            # Train locally
            client_model, loss = train_local_model(
                client_model,
                client_data[client_idx],
                local_epochs,
                lr
            )

            client_models.append(client_model)
            train_losses.append(loss)

            if (i + 1) % 3 == 0 or (i + 1) == num_selected:
                print(f"  Clients {i-1+1}-{i+1}: Avg loss = {np.mean(train_losses[-3:]):.4f}")

        avg_train_loss = np.mean(train_losses)
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        history['train_loss'].append(avg_train_loss)

        # Step 3: Server-Side Clustering
        print("\n[Server-Side Clustering]")
        clusters, cluster_labels = cluster_clients(client_models, num_clusters)

        print(f"Cluster assignments:")
        for cluster_id, clients in clusters.items():
            print(f"  Cluster {cluster_id + 1}: {len(clients)} clients - {clients}")

        history['cluster_assignments'].append(cluster_labels.tolist())

        # Step 4: Cluster Aggregation
        print("\n[Cluster Aggregation]")
        cluster_models = {}
        cluster_encoders = {}
        cluster_predictors = {}

        for cluster_id, cluster_client_indices in clusters.items():
            if len(cluster_client_indices) == 0:
                continue

            # Aggregate models in this cluster
            aggregated_model = aggregate_cluster_models(
                client_models,
                cluster_client_indices,
                [client_data[selected_clients[i]] for i in range(len(selected_clients))]
            )

            cluster_models[cluster_id] = aggregated_model
            cluster_encoders[cluster_id] = aggregated_model.get_encoder_params()
            cluster_predictors[cluster_id] = aggregated_model.get_predictor_params()

            print(f"  Cluster {cluster_id + 1}: Aggregated {len(cluster_client_indices)} models")

        # Step 5: Update global model (use first cluster model as reference)
        if len(cluster_models) > 0:
            global_model = deepcopy(cluster_models[0])

        # Step 6: Client Adaptation (simulate on selected clients)
        print("\n[Client Adaptation]")
        adapted_models = []

        for i, client_idx in enumerate(selected_clients[:3]):  # Adapt first 3 for demo
            cluster_id = cluster_labels[i]

            adapted_model = adapt_client_model(
                client_idx,
                cluster_id,
                cluster_encoders,
                cluster_predictors[cluster_id],
                client_data[client_idx],
                adaptation_epochs,
                lr
            )

            adapted_models.append(adapted_model)
            print(f"  Client {client_idx}: Adapted to cluster {cluster_id + 1}")

        # Evaluation on test set using global model
        test_mse, test_mae = evaluate_model(global_model, X_test_tensor, y_test_tensor)
        history['test_mse'].append(test_mse)
        history['test_mae'].append(test_mae)

        print(f"\n[Evaluation]")
        print(f"  Test MSE: {test_mse:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")

    return global_model, cluster_models, history

# ============================================================================
# Step 9: Run MCFL Algorithm
# ============================================================================
print("\n[Step 9] Running MCFL algorithm...")

# Hyperparameters
num_rounds = 10
num_clusters = 3
local_epochs = 5
adaptation_epochs = 3
learning_rate = 0.001
client_selection_ratio = 0.6

# Run MCFL
final_model, cluster_models, history = run_mcfl(
    num_rounds=num_rounds,
    num_clusters=num_clusters,
    local_epochs=local_epochs,
    adaptation_epochs=adaptation_epochs,
    lr=learning_rate,
    client_selection_ratio=client_selection_ratio
)

# ============================================================================
# Step 10: Visualize Results
# ============================================================================
print("\n" + "=" * 80)
print("Visualizing Results")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Training Loss
axes[0, 0].plot(range(1, num_rounds + 1), history['train_loss'], 'b-o', linewidth=2)
axes[0, 0].set_xlabel('Round', fontsize=12)
axes[0, 0].set_ylabel('Training Loss', fontsize=12)
axes[0, 0].set_title('MCFL Training Loss Over Rounds', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Test MSE
axes[0, 1].plot(range(1, num_rounds + 1), history['test_mse'], 'r-s', linewidth=2)
axes[0, 1].set_xlabel('Round', fontsize=12)
axes[0, 1].set_ylabel('Test MSE', fontsize=12)
axes[0, 1].set_title('MCFL Test MSE Over Rounds', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Test MAE
axes[1, 0].plot(range(1, num_rounds + 1), history['test_mae'], 'g-^', linewidth=2)
axes[1, 0].set_xlabel('Round', fontsize=12)
axes[1, 0].set_ylabel('Test MAE', fontsize=12)
axes[1, 0].set_title('MCFL Test MAE Over Rounds', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Cluster Evolution
cluster_evolution = np.array(history['cluster_assignments']).T
for client_idx in range(min(6, len(cluster_evolution))):  # Show first 6 clients
    axes[1, 1].plot(range(1, num_rounds + 1), cluster_evolution[client_idx],
                   '-o', label=f'Client {client_idx}', linewidth=2)

axes[1, 1].set_xlabel('Round', fontsize=12)
axes[1, 1].set_ylabel('Cluster ID', fontsize=12)
axes[1, 1].set_title('Client Cluster Assignments Over Rounds', fontsize=14, fontweight='bold')
axes[1, 1].legend(loc='best', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_yticks(range(num_clusters))

plt.tight_layout()
plt.savefig('mcfl_results.png', dpi=300, bbox_inches='tight')
print("\nResults saved to 'mcfl_results.png'")

# Print final statistics
print("\n" + "=" * 80)
print("Final Results Summary")
print("=" * 80)
print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
print(f"Final Test MSE: {history['test_mse'][-1]:.4f}")
print(f"Final Test MAE: {history['test_mae'][-1]:.4f}")
print(f"\nBest Test MSE: {min(history['test_mse']):.4f} (Round {np.argmin(history['test_mse']) + 1})")
print(f"Best Test MAE: {min(history['test_mae']):.4f} (Round {np.argmin(history['test_mae']) + 1})")

print("\n" + "=" * 80)
print("MCFL Algorithm Completed Successfully!")
print("=" * 80)
