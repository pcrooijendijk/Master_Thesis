from utils import Dataset

dataset = Dataset("/content/drive/MyDrive/data_DocBench")
dataset.convert_to_json(0, "test_documents.json")