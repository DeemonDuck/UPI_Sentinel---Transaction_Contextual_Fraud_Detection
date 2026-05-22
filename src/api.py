from fastapi import APIRouter
from pydantic import BaseModel
from src.model_loader import load_artifacts
from src.sequence_builder import SequenceBuilder
from src.inference import predict_transaction

router = APIRouter()

user_buffers = {}

# Load everything once (not inside function)

base_path = "artifacts"   

model, scaler, label_encoder, metadata = load_artifacts(base_path)

sequence_length = metadata["sequence_length"]
seq_builder = SequenceBuilder(sequence_length)
print("MODEL:", type(model))
print("SCALER:", type(scaler))
print("ENCODER:", type(label_encoder))

class TransactionRequest(BaseModel):
    user_id: str
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

    class Config:
        extra = "ignore"


@router.get("/")
def home():
    return {"message": "API is working "}


@router.post("/predict")
def predict(request: TransactionRequest):

    data = request.model_dump()

    user_id = data.pop("user_id")   # remove before model
    txn = data

    if user_id not in user_buffers:
        user_buffers[user_id] = SequenceBuilder(sequence_length)

    seq_builder = user_buffers[user_id]

    result = predict_transaction(
        txn,
        model,
        scaler,
        label_encoder,
        seq_builder,
        sequence_length
    )

    return result