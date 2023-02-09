import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import get_MCM_score
    
def train(train_loader, device, model, criterion, optimizer, log_file = None):
    model.train()
    num_batch = len(train_loader)
    model = model.to(device)
    train_loss = 0.0
    all_labs = []
    all_preds = []
    
    for i, data in enumerate(train_loader, 0):
        baseline_inputs, labels, bg_inputs, batch_seq_len,  = data[0].to(device), data[1].to(device), data[2].to(device), data[3]
        optimizer.zero_grad()
        outputs = model(baseline_inputs, bg_inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        _,pred = outputs.topk(1)

        train_loss += loss.item()
        for mm in labels:
            all_labs.append(mm.item())
        for mm in pred:
            all_preds.append(mm.item())
    
    scores = get_MCM_score(all_labs, all_preds) 
    train_loss/= num_batch
    if log_file:
        log_file.write(f'Train Loss: {train_loss:>0.4f}')
        log_file.write(f'Train: F1 {(100*scores[0]):>0.2f}%, P {(100*scores[1]):>0.2f}%, R {(100*scores[2]):>0.2f}%, ACC {(100*scores[3]):>0.2f}%\n')
    return train_loss

def test(validate_loader, device, model, criterion, is_valid=False, log_file = None):
    model = model.to(device)
    model.eval()
    num_batch = len(validate_loader)
    model = model.to(device)
    test_loss = 0.0
    all_labs = []
    all_preds = []
    for i, data in enumerate(validate_loader, 0):
        baseline_inputs, labels, bg_inputs, batch_seq_len,  = data[0].to(device), data[1].to(device), data[2].to(device), data[3]
        outputs = model(baseline_inputs, bg_inputs)
        loss = criterion(outputs,labels)
        _,pred = outputs.topk(1)

        test_loss += loss.item()
        for mm in labels:
            all_labs.append(mm.item())
        for mm in pred:
            all_preds.append(mm.item())
    
    scores = get_MCM_score(all_labs, all_preds) 
    test_loss /= num_batch
    if log_file:        
        if is_valid:
            log_file.write(f'Valid: F1 {(100*scores[0]):>0.2f}%, P {(100*scores[1]):>0.2f}%, R {(100*scores[2]):>0.2f}%, ACC {(100*scores[3]):>0.2f}%\n')
        else:
            log_file.write(f'Test : F1 {(100*scores[0]):>0.2f}%, P {(100*scores[1]):>0.2f}%, R {(100*scores[2]):>0.2f}%, ACC {(100*scores[3]):>0.2f}%\n')
    return scores