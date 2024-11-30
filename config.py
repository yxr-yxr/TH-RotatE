# Configuration for the TH-RotatE model hyperparameters

# General settings
BATCH_SIZE = 1024              # Number of samples per batch
HIDDEN_DIM = 700               # Dimensionality of the hidden layer
ALPHA = 1.0                    # Margin for positive-negative triples
ADVERSARIAL_TEMPERATURE = 1.0  # Temperature for adversarial sampling
LEARNING_RATE = 0.001          # Learning rate for the optimizer
GAMMA = 24.0                   # Margin value for pairwise ranking loss
TEST_BATCH_SIZE = 10           # Batch size during testing
MAX_STEPS = 100000             # Maximum number of training iterations

# Negative sampling settings
NEGATIVE_TRIPLES_PER_POSITIVE = 64  # Number of negative triples per positive triple
NEGATIVE_SAMPLING_METHOD = "dynamic and adversarial sampling approach"

# Print out the hyperparameters for confirmation
def print_hyperparameters():
    print("TH-RotatE Model Hyperparameters:")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Hidden_dim: {HIDDEN_DIM}")
    print(f"Alpha: {ALPHA}")
    print(f"Adversarial_temperature: {ADVERSARIAL_TEMPERATURE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Gamma: {GAMMA}")
    print(f"Test batch size: {TEST_BATCH_SIZE}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Negative triples per positive: {NEGATIVE_TRIPLES_PER_POSITIVE}")
    print(f"Negative sampling method: {NEGATIVE_SAMPLING_METHOD}")

# Call to print hyperparameters
if __name__ == "__main__":
    print_hyperparameters()
