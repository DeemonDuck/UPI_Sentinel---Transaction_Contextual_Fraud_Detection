import numpy as np
from src.preprocessing import preprocess_transaction


def predict_transaction(txn, model, scaler, label_encoder, seq_builder, sequence_length):

    # Step 1: Preprocess
    processed = preprocess_transaction(txn, scaler, label_encoder)

    # Step 2: Update sequence
    seq = seq_builder.update(processed)

    # Step 3: Check if enough transactions
    if not seq_builder.is_ready():
        return {"message": "Waiting for more transactions"}

    # Step 4: Reshape for model
    seq = seq.reshape(1, sequence_length, seq.shape[1])

    # Step 5: Predict
    prob = model.predict(seq, verbose=0)[0][0]

    # Step 6: Risk logic
    if prob < 0.5:
        risk = "SAFE"
    elif prob < 0.9:
        risk = "SUSPICIOUS"
    else:
        risk = "FRAUD"

    return {
        "fraud_probability": float(prob),
        "risk_level": risk
    }