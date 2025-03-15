import torch

class Config:
    PROJECT_ID="true-sprite-412217"
    BUCKET_NAME="pleasedontbankrupt"
    OUTPUT_DIR="bert_checkpoints"
    DATASET_PATH="sudarshan1927/go-emotions-and-generated"
    RANDOM_STATE=42
    MODEL_PATH="nlptown/bert-base-multilingual-uncased-sentiment"
    NUM_LABELS=28
    PROJECT_NAME="sst_with_bert"
    
    @classmethod
    def device(cls):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def set_roberta_model(cls):
        cls.MODEL_PATH  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        cls.OUTPUT_DIR = "roberta_checkpoints_cls"
    
    @classmethod
    def set_bert_model(cls):
        cls.MODEL_PATH = "nlptown/bert-base-multilingual-uncased-sentiment"
        cls.OUTPUT_DIR = "bert_checkpoints_cls"
