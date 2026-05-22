import sys
import os

# Add src to path
sys.path.append(os.path.abspath("../src"))

from src.model_loader import load_artifacts
from src.sequence_builder import SequenceBuilder
from simulate_stream import simulate_transaction_stream


# Step 1: Load artifacts
BASE_PATH = "../artifacts"

model, scaler, label_encoder, metadata = load_artifacts(BASE_PATH)

SEQUENCE_LENGTH = metadata["sequence_length"]


# Step 2: Initialize sequence builder
seq_builder = SequenceBuilder(SEQUENCE_LENGTH)


# Step 3: Sample transactions
transactions = [
    {
        "step": 1,
        "type": "TRANSFER",
        "amount": 2000,
        "oldbalanceOrg": 5000,
        "newbalanceOrig": 3000,
        "oldbalanceDest": 1000,
        "newbalanceDest": 3000
    },
    {
        "step": 2,
        "type": "TRANSFER",
        "amount": 1500,
        "oldbalanceOrg": 3000,
        "newbalanceOrig": 1500,
        "oldbalanceDest": 3000,
        "newbalanceDest": 4500
    },
    {
        "step": 3,
        "type": "TRANSFER",
        "amount": 1200,
        "oldbalanceOrg": 1500,
        "newbalanceOrig": 300,
        "oldbalanceDest": 4500,
        "newbalanceDest": 5700
    },
    {
        "step": 4,
        "type": "TRANSFER",
        "amount": 250,
        "oldbalanceOrg": 300,
        "newbalanceOrig": 50,
        "oldbalanceDest": 5700,
        "newbalanceDest": 5950
    },
    {
        "step": 5,
        "type": "PAYMENT",
        "amount": 10,
        "oldbalanceOrg": 50,
        "newbalanceOrig": 40,
        "oldbalanceDest": 5950,
        "newbalanceDest": 5960
    }
]


# Step 4: Run simulation
simulate_transaction_stream(
    transactions,
    model,
    scaler,
    label_encoder,
    seq_builder,
    SEQUENCE_LENGTH
)