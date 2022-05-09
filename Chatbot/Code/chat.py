import torch
from model import BERT_Arch
import numpy as np
import re
import random

#Load previous parameters
savePath="modelData.pth"
data=torch.load(savePath)

model_state=data["model_state"]
k_classes=data["k_classes"]
responses=data["responses"]
tags=data["tags"]
max_seq_len=data["max_seq_len"]
print("Data loaded")
# Initialize new model
model = BERT_Arch(k_classes)
model = model.to(model.device)
print("New model created")
#Load previous parameters from saved data
model.load_state_dict(model_state)
print("Model parameters correctly loaded")
bot_name="Bart"

#Gets prediction from the model
def get_prediction(str, pModel):
    str = re.sub(r"[^a-zA-Z ]+", "",str)
    test_text = [str]
    pModel.eval()

    tokens_test_data = pModel.tokenizer(
        test_text,
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data["input_ids"])
    test_mask = torch.tensor(tokens_test_data["attention_mask"])

    preds = None
    with torch.no_grad():
        preds = pModel(test_seq.to(pModel.device), test_mask.to(pModel.device))
        preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    if(True):
        return tags[preds[0]]
    else:
        return "unknown"

def get_response(message, pModel):
    intent = get_prediction(message, pModel)
    if(intent!="unknown"):
        print(f"Response : {intent}")
        rand_idx = random.randrange(len(responses[intent]))
        return responses[intent][rand_idx]
    else:
        return "I did not understand"

def get_model():
    return model


"""
print("Let's chat, type 'quit' to exit")
while True:
    sentence=input('You: ')
    if(sentence=='quit'):
        break

    print(f"{bot_name}: {get_response(sentence,model)}")
"""
