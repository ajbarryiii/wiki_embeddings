import torch
import torch.nn as nn
import nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from typing import Optional, Any, List, Dict
from datasets import IterableDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from pathlib import Path
import numpy as np
import json
from datetime import datetime

class MLMTrainer:
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        dataset: IterableDataset,
        tokenizer: Any,
        checkpoint_dir: str,
        log_dir: str,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        max_length: int = 128,
        device: Optional[str] = None,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
        self.dataset = dataset
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
            return_tensors="pt"
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Add logging attributes
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize loss tracking
        self.train_losses = []
        self.running_losses = []
        self.epochs = []
        self.steps = []

        # Add checkpoint attributes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint settings
        self.checkpoint_interval = 5000  # Save every 5000 steps
        self.max_checkpoints = 3  # Keep only the last 3 checkpoints

    def save_checkpoint(self, step, epoch, loss, optimizer):
        """Save a checkpoint of the model"""
        try:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_{step:07d}'
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save models
            torch.save(self.encoder.state_dict(), 
                      checkpoint_path / 'encoder.pt')
            torch.save(self.decoder.state_dict(), 
                      checkpoint_path / 'decoder.pt')
            
            # Save optimizer state
            torch.save(optimizer.state_dict(), 
                      checkpoint_path / 'optimizer.pt')
            
            # Save training state
            training_state = {
                'step': step,
                'epoch': epoch,
                'loss': loss,
                'learning_rate': self.learning_rate,
                'train_losses': self.train_losses,
                'running_losses': self.running_losses,
                'steps': self.steps,
                'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            }
            
            with open(checkpoint_path / 'training_state.json', 'w') as f:
                json.dump(training_state, f, indent=2)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            print(f"\nCheckpoint saved at step {step}")
            return True
            
        except Exception as e:
            print(f"Error saving checkpoint at step {step}: {str(e)}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """Keep only the most recent checkpoints"""
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x.name.split('_')[1])
        )
        
        while len(checkpoints) > self.max_checkpoints:
            checkpoint_to_remove = checkpoints.pop(0)
            try:
                for file in checkpoint_to_remove.iterdir():
                    file.unlink()
                checkpoint_to_remove.rmdir()
            except Exception as e:
                print(f"Error removing old checkpoint: {str(e)}")

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        try:
            # Load models
            self.encoder.load_state_dict(
                torch.load(checkpoint_path / 'encoder.pt')
            )
            self.decoder.load_state_dict(
                torch.load(checkpoint_path / 'decoder.pt')
            )
            
            # Load training state
            with open(checkpoint_path / 'training_state.json', 'r') as f:
                training_state = json.load(f)
            
            # Restore training state
            self.train_losses = training_state['train_losses']
            self.running_losses = training_state['running_losses']
            self.steps = training_state['steps']
            
            print(f"Loaded checkpoint from step {training_state['step']}")
            return training_state
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return None

    def find_latest_checkpoint(self):
        """Find the most recent checkpoint"""
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x.name.split('_')[1])
        )
        return checkpoints[-1] if checkpoints else None


    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Improved collation function with proper batch handling"""
        try:
            
            # Process each text in the batch
            processed_batch = []
            for item in batch:
                # Handle different input formats
                text = item["text"] if isinstance(item, dict) else item
                
                # Tokenize the text
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors=None  # Don't add batch dimension yet
                )
                
                # Add to processed batch
                processed_batch.append({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                })
            
            # Apply MLM using the collator
            mlm_batch = self.collator(processed_batch)
            
            return mlm_batch
            
        except Exception as e:
            print(f"Error in collate_fn: {str(e)}")
            print(f"Batch size: {len(batch)}")
            raise

    def prepare_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=False  # Keep partial batches
        )

    def plot_losses(self, save=True, show=False):
        """Plot training losses"""
        plt.figure(figsize=(10, 6))
        
        # Plot running average loss
        if self.running_losses:
            plt.plot(self.steps, self.running_losses, 
                    label='Running Loss', alpha=0.3, color='b')
        
        # Plot epoch average loss
        if self.train_losses:
            plt.plot(np.linspace(1, len(self.train_losses), len(self.train_losses)),
                    self.train_losses, label='Epoch Loss', 
                    marker='o', color='r')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.log_dir / 'loss_plot.png')
        if show:
            plt.show()
        plt.close()

    def train(self):
        dataloader = self.prepare_dataloader()
        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.learning_rate
        )
        
        log_interval = 100  # Update progress bar every 100 iterations
        total_steps = 0
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.decoder.train()
            epoch_loss = 0.0
            running_loss = 0.0
            
            progress_bar = tqdm(enumerate(dataloader), 
                              desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                              mininterval=10.0)
            
            for step, batch in progress_bar:
                try:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    # Forward pass through encoder
                    encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
                    
                    # Forward pass through decoder
                    logits = self.decoder(
                        x=input_ids,
                        encoder_out=encoder_outputs,
                        attention_mask=attention_mask
                    )
                    
                    # Ensure shapes match
                    batch_size, seq_length = input_ids.shape
                    if logits.shape[1] != seq_length:
                        if logits.shape[1] < seq_length:
                            pad_size = seq_length - logits.shape[1]
                            logits = F.pad(logits, (0, 0, 0, pad_size))
                        else:
                            logits = logits[:, :seq_length, :]
                    
                    # Compute loss
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)
                    loss = self.criterion(logits, labels)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update loss tracking
                    current_loss = loss.item()
                    epoch_loss += current_loss
                    running_loss += current_loss
                    total_steps += 1
                    
                    # Update progress bar and plot losses periodically
                    if (step + 1) % log_interval == 0:
                        avg_running_loss = running_loss / log_interval
                        self.running_losses.append(avg_running_loss)
                        self.steps.append(total_steps)
                        
                        # Update progress bar
                        elapsed = time.time() - start_time
                        progress_bar.set_postfix({
                            "Loss": f"{avg_running_loss:.4f}",
                            "Steps/sec": f"{total_steps/elapsed:.2f}"
                        })
                        
                        # Plot current progress
                        if total_steps % (log_interval * 5) == 0:  # Plot every 500 steps
                            self.plot_losses()  # Save but don't show
                        
                        running_loss = 0.0

                    # Checkpoint saving
                    if total_steps % self.checkpoint_interval == 0:
                        self.save_checkpoint(
                            step=total_steps,
                            epoch=epoch,
                            loss=current_loss,
                            optimizer=optimizer
                        )

                    
                except Exception as e:
                    print(f"\nError in training step {step}:")
                    print(f"Input shapes:")
                    print(f"- input_ids: {input_ids.shape}")
                    print(f"- attention_mask: {attention_mask.shape}")
                    print(f"- labels: {labels.shape}")
                    raise e
            
            # End of epoch logging
            avg_epoch_loss = epoch_loss / (total_steps//epoch)
            self.train_losses.append(avg_epoch_loss)
            self.epochs.append(epoch + 1)
            
            print(f"\nEpoch [{epoch + 1}/{self.num_epochs}] "
                  f"Complete - Average Loss: {avg_epoch_loss:.4f}")
            
            # Plot and save after each epoch
            self.plot_losses(show=False)
        
        # Final plots
        self.plot_losses(show=True)
        self.save_training_history()
    
    def save_training_history(self):
        """Save training history to files"""
        history = {
            'steps': self.steps,
            'running_losses': self.running_losses,
            'epochs': self.epochs,
            'train_losses': self.train_losses,
        }
        
        np.save(self.log_dir / 'training_history.npy', history)
        
        # Save as CSV for easy analysis
        import pandas as pd
        df = pd.DataFrame({
            'step': self.steps,
            'running_loss': self.running_losses
        })
        df.to_csv(self.log_dir / 'training_history.csv', index=False)


    def debug_batch(self, batch):
        """Helper function to debug batch contents"""
        print("\nDebugging batch:")
        print(f"Batch type: {type(batch)}")
        print(f"Batch size: {len(batch)}")
        print("First item:")
        first_item = batch[0]
        print(f"Type: {type(first_item)}")
        if isinstance(first_item, dict):
            for key, value in first_item.items():
                print(f"{key}: {type(value)}")
                if isinstance(value, str):
                    print(f"Text preview: {value[:100]}...")
