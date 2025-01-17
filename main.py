from transformers import AutoTokenizer
from datasets import load_dataset, IterableDataset

from model import Encoder, Decoder
from trainer import MLMTrainer

def main():
    # Initialize the encoder and decoder 
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = Encoder(vocab_size=tokenizer.vocab_size, embed_size=768, num_heads=12, forward_expansion=4, dropout=0.1, num_layers=6, latent_dim=128)
    decoder = Decoder(vocab_size=tokenizer.vocab_size, embed_size=768, num_heads=12, forward_expansion=4, dropout=0.1, num_layers=6, latent_dim=128)

    # Load the Wikipedia dataset in streaming mode
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code = True)

    # Create a Trainer instance
    trainer = MLMTrainer(
        encoder=encoder,
        decoder=decoder,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=8,
        learning_rate=1e-5,
        num_epochs=3,
        max_length=128,
        device = "mps",
        checkpoint_dir="model_checkpoints",
        log_dir="log_checkpoints"
    )

    # Train the model
    trainer.train()

if __name__=="__main__": 
    main()
