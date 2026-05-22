import pandas as pd

def preprocess_transaction(txn, scaler, label_encoder):  

    df = pd.DataFrame([txn])

    # Encode transaction type
    df['type'] = label_encoder.transform(df['type'])

    # Numeric columns
    num_cols = [
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest'
    ]

    df[num_cols] = scaler.transform(df[num_cols])

    # Final feature order 
    features = [
        'type',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest'
    ]

    return df[features].values