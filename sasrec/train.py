import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sasrec.seed import set_seed
from sasrec.sasrecnet import SASRecNet
from sasrec.sasrecdataset import SASRecDataset
from sasrec.preprocess import preprocess
from evaluate.evaluate import evaluate_model

MAX_LEN = 300
HIDDEN_SIZE = 128
NUM_HEADS = 2
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
PATIENCE_LIMIT = 10
DATA_PATH = r"D:\HUST\2025.2\ML\Project\data\ratings.csv"
GENRE_PATH = r"D:\HUST\2025.2\ML\Project\data\movies.csv"
CHECKPOINT = "sasrec_checkpoint.pth"
EPOCHS = 100
LEARNING_RATE = 0.002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SASRec():
    def __init__(self, data=DATA_PATH, max_len=MAX_LEN, checkpoint_path=CHECKPOINT, 
                 patience_lim=PATIENCE_LIMIT, epochs=EPOCHS, lr=LEARNING_RATE, 
                 random_seed=1, batch_size=32, best_weights_path=None):
        set_seed(random_seed)
        seqs, num_items, self.item_map, self.movie_to_genre, self.num_genres = preprocess(data, GENRE_PATH)
        
        train_seqs = [seq[:-2] for seq in seqs]
        eval_seqs = [seq[:-1] for seq in seqs]

        train_dataset = SASRecDataset(train_seqs, self.movie_to_genre, max_len, is_training=True, stride=None)
        eval_dataset = SASRecDataset(eval_seqs, self.movie_to_genre, max_len)

        pin_mem = DEVICE.type == "cuda"
        num_w = 4 if DEVICE.type == "cuda" else 0

        self.dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_w, pin_memory=pin_mem)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_w, pin_memory=pin_mem)

        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.best_weights_path = best_weights_path if best_weights_path else f"sasrec_BEST_weights_{random_seed}.pth"
        self.best_epoch = 0
        self.patience_lim = patience_lim
        self.lr = lr
        self.best_eval_ndcg = 0.0  # Initialized at 0.0 since higher NDCG is better!

    def train(self, **params):
        print(f"Training on {DEVICE}")
        
        model = SASRecNet(
            num_items=params['num_items'], 
            max_len=params["max_len"], 
            hidden_size=params["hidden_size"], 
            num_heads=params["num_heads"],
            dropout_rate=params["dropout_rate"],
            num_genres=params["num_genres"]
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(ignore_index=0) 
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # OneCycleLR Scheduler for Transformer Warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.lr, 
            steps_per_epoch=len(self.dataloader), 
            epochs=self.epochs, 
            pct_start=0.2 # Spend the first 10% of training warming up the learning rate
        )

        start_epoch = 0
        patience_counter = 0

        if os.path.exists(self.checkpoint_path):
            print(f"Found checkpoint at {self.checkpoint_path}. Resuming training...")
            checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            self.best_eval_ndcg = checkpoint.get('best_eval_ndcg', 0.0)
            patience_counter = checkpoint.get('patience_counter', 0)
            self.best_epoch = checkpoint.get('best_epoch', 0)
            print(f"Successfully resumed. Starting from Epoch {start_epoch + 1}")
        else:
            print("No checkpoint found. Starting from Epoch 1.")

        for epoch in range(start_epoch, self.epochs):
            model.train()
            total_loss = 0.0
    
            for batch_inputs, batch_targets, batch_genres in self.dataloader:
                batch_inputs = batch_inputs.to(DEVICE, non_blocking=True)
                batch_targets = batch_targets.to(DEVICE, non_blocking=True)
                batch_genres = batch_genres.to(DEVICE, non_blocking=True)
        
                optimizer.zero_grad(set_to_none=True)
        
                seq_output = model(batch_inputs, batch_genres) 
                logits = torch.matmul(seq_output, model.item_emb.weight.t())
                logits = logits.permute(0, 2, 1) 
                loss = criterion(logits, batch_targets)
        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                scheduler.step()
        
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(self.dataloader)
            
            # Step into metric verification using Leave-One-Out (LOO) with past history masking
            val_hr, val_ndcg, val_mrr = evaluate_model(model, self.eval_dataloader, DEVICE, k=10)
        
            # Early stopping check based directly on performance metric ceiling
            if val_ndcg > self.best_eval_ndcg:  
                self.best_eval_ndcg = val_ndcg
                self.best_epoch = epoch + 1
                torch.save(model.state_dict(), self.best_weights_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Epoch [{epoch + 1}/{self.epochs}] | Train Loss: {avg_train_loss:.4f} | Val HR@10: {val_hr:.4f} | Val NDCG@10: {val_ndcg:.4f}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_eval_ndcg': self.best_eval_ndcg,
                'patience_counter': patience_counter,
                'best_epoch': self.best_epoch,
                'loss': loss.item()
            }
            torch.save(checkpoint, self.checkpoint_path)

            if patience_counter >= self.patience_lim:
                print("Early stopping triggered. Progress halted.")
                break

        print(f"Training Complete. Optimal model weights at epoch {self.best_epoch} preserved. Best Validation NDCG@10: {self.best_eval_ndcg:.4f}")

if __name__ == "__main__":
    seqs, num_items, item_map, movie_to_genre, genres_num = preprocess(DATA_PATH, GENRE_PATH)
    print(f"Total processed movie items in catalog: {num_items}")
    
    hyperparams = {
        "num_items": num_items,
        "max_len": MAX_LEN,
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "dropout_rate": DROPOUT_RATE,
        "learning_rate": LEARNING_RATE,
        "num_genres":genres_num
    }

    seeds = [1282]
    for seed in seeds:
        print(f"\n--- Running Training Execution with Seed {seed} ---")
        checkpoint_name = f"sasrec_checkpoint_seed_{seed}.pth"
        weights_name = f"sasrec_BEST_weights_{seed}.pth"
        
        runner = SASRec(
            data=DATA_PATH,
            max_len=MAX_LEN,
            checkpoint_path=checkpoint_name,
            best_weights_path=weights_name,
            random_seed=seed,
            lr=hyperparams["learning_rate"],
            batch_size=BATCH_SIZE
        )
        runner.train(**hyperparams)