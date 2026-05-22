from src.inference import predict_transaction


def simulate_transaction_stream(transactions, model, scaler, label_encoder, seq_builder, sequence_length):

    print("Starting Transaction Stream...\n")

    for i, txn in enumerate(transactions, start=1):

        result = predict_transaction(
            txn,
            model,
            scaler,
            label_encoder,
            seq_builder,
            sequence_length
        )

        print(f"Transaction {i}")
        print("Input:", txn)
        print("Output:", result)
        print("-" * 50)