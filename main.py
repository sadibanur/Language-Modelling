import torch
from data import GPTTokenizedData
from model import get_best_model_definition
from train import train_model
from evaluation import perplexity


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get dataloaders (data.py)
    tokenized = GPTTokenizedData()
    vocab_size = tokenized.vocab_size

    # all 3 dataloaders in a dictionary with keys 'train', 'test', 'val 
    dataloaders = tokenized.dataloaders 
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # instantiate model (model.py)
    model = get_best_model_definition(vocab_size=vocab_size)
    model.to(device)

    # train model (train.py)
    trained_model = train_model(model, train_loader, val_loader, device, vocab_size)

    model = get_best_model_definition(vocab_size)
    model.load_state_dict(torch.load('best_model.pt'))
    model.to(device)

    # evaluate perplexity for all three splits (evaluate.py)
    model.eval()
    
    # Train loss, perplexity
    train_ppl, train_loss = perplexity(model, dataloaders['train'])
    print(f"Training perplexity: {train_ppl:.2f}")
    print(f"Train Loss: {train_loss:.4f}")
    
    # Validation loss, perplexity
    val_ppl, val_loss = perplexity(model, dataloaders['val'])
    print(f"Validation perplexity: {val_ppl:.2f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Test loss, perplexity
    test_ppl, test_loss = perplexity(model, dataloaders['test'])
    print(f"Test perplexity: {test_ppl:.2f}")
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":

    main()
