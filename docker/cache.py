import transformers
text_tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
text_encoder = transformers.AutoModel.from_pretrained("roberta-base")
text_tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
text_encoder = transformers.AutoModel.from_pretrained("microsoft/deberta-v3-base")
