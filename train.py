import torch
from evaluation import perplexity


"""
TODO write your training loop here.
Things to take care with:
    - make sure to use the correct loss function for the task
    - make sure that the targets are correct (each token should predict the next token in the sequence)
    - there should be no loss for padding tokens.
"""

def train_model(model, train_loader, val_loader, device, vocab_size, save_path='best_model.pt'):
    "Train the Transformer model for predicting the next-token"

    # Ignore padding tokens
    ignore_idx = -100

    best_loss = float('inf')
    max_grad_norm = 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_idx)

    # Loop over 8 epochs 
    for epoch in range(8):
        model.train()

        # Training 
        for batch in train_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            inputs = input_ids[:, :-1].to(device)
            targets = input_ids[:, 1:].to(device)

            padding_mask_input = attention_mask[:, :-1].to(device)
            padding_mask_target = attention_mask[:, 1:].to(device)

            targets = targets.masked_fill(padding_mask_target==0, ignore_idx).view(-1)

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs, padding_mask_input)

            B, S, V = logits.shape
            
            logits = logits.view(-1, V)

            loss = loss_fn(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # Validation
        model.eval()

        val_ppl, val_loss = perplexity(model, val_loader)

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
        
    return model
    
