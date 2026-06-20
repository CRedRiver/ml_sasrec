import os
import argparse
import json
import csv
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
STRIDE = 100
DROPOUT_RATE = 0.3
BATCH_SIZE = 256    
PATIENCE_LIMIT = 10
DATA_PATH = r"D:\HUST\2025.2\ML\Project\data\ratings.csv"
GENRE_PATH = r"D:\HUST\2025.2\ML\Project\data\movies.csv"
CHECKPOINT = "sasrec_checkpoint.pth"
EPOCHS = 100
LEARNING_RATE = 0.002
NEG_SAMP=2000
SMOOTH = 0.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SASRec():
    def __init__(self, data=DATA_PATH, genre_path=GENRE_PATH, max_len=MAX_LEN, checkpoint_path=CHECKPOINT, 
                 patience_lim=PATIENCE_LIMIT, epochs=EPOCHS, lr=LEARNING_RATE, 
                 random_seed=1, batch_size=32, best_weights_path=None, out_dir=None):
        set_seed(random_seed)
        self.random_seed = random_seed
        self.out_dir = out_dir if out_dir is not None else os.getcwd()
        os.makedirs(self.out_dir, exist_ok=True)

        seqs, num_items, self.item_map, self.movie_to_genre, self.num_genres = preprocess(data, genre_path)
        
        # remove last items for train set (remove 2 last) and validation set (remove 1 last)
        train_seqs = [seq[:-2] for seq in seqs]
        eval_seqs = [seq[:-1] for seq in seqs]

        train_dataset = SASRecDataset(train_seqs, self.movie_to_genre, max_len, is_training=True, stride=STRIDE)
        eval_dataset = SASRecDataset(eval_seqs, self.movie_to_genre, max_len)

        pin_mem = DEVICE.type == "cuda"
        num_w = 2 if DEVICE.type == "cuda" else 0

        self.dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_w, pin_memory=pin_mem)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_w, pin_memory=pin_mem)

        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.best_weights_path = best_weights_path if best_weights_path else f"sasrec_BEST_weights_{random_seed}.pth"
        self.best_epoch = 0
        self.patience_lim = patience_lim
        self.lr = lr
        self.best_eval_ndcg = 0.0  

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

        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0) 
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # OneCycleLR Scheduler for Transformer Warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.lr, 
            steps_per_epoch=len(self.dataloader), 
            epochs=self.epochs, 
            pct_start=0.1
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

        metrics_history = []

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
            val_hr, val_ndcg, val_mrr = evaluate_model(model, self.eval_dataloader, DEVICE, k=10, neg_samp=NEG_SAMP)
            # record metrics for this epoch (train loss, ndcg, hr, mrr)
            try:
                metrics_history.append({
                    "epoch": epoch,
                    "train_loss": float(avg_train_loss) if avg_train_loss is not None else None,
                    "ndcg": float(val_ndcg) if val_ndcg is not None else None,
                    "hr": float(val_hr) if val_hr is not None else None,
                    "mrr": float(val_mrr) if val_mrr is not None else None
                })
            except Exception:
                metrics_history.append({"epoch": epoch, "train_loss": None, "ndcg": None, "hr": None, "mrr": None})
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

        # persist metrics log (JSON + CSV) for later plotting
        json_name = os.path.join(self.out_dir, f"metrics_log_seed_{getattr(self, 'random_seed', 'unknown')}.json")
        csv_name = os.path.join(self.out_dir, f"metrics_log_seed_{getattr(self, 'random_seed', 'unknown')}.csv")
        try:
            with open(json_name, 'w') as f:
                json.dump({"metrics_history": metrics_history, "best_eval_ndcg": self.best_eval_ndcg, "best_epoch": self.best_epoch}, f)
            print(f"Saved metrics JSON to {json_name}")
        except Exception as e:
            print(f"Warning: failed to save metrics JSON: {e}")

        try:
            with open(csv_name, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "ndcg", "hr", "mrr"])
                writer.writeheader()
                for row in metrics_history:
                    writer.writerow({
                        "epoch": row.get("epoch"),
                        "train_loss": row.get("train_loss"),
                        "ndcg": row.get("ndcg"),
                        "hr": row.get("hr"),
                        "mrr": row.get("mrr")
                    })
            print(f"Saved metrics CSV to {csv_name}")
        except Exception as e:
            print(f"Warning: failed to save metrics CSV: {e}")

        print(f"Training Complete. Optimal model weights at epoch {self.best_epoch} preserved. Best Validation NDCG@10: {self.best_eval_ndcg:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SASRec model with CLI options")
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--genre_path", type=str, default=GENRE_PATH)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num_heads", type=int, default=NUM_HEADS)
    parser.add_argument("--dropout_rate", type=float, default=DROPOUT_RATE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience_lim", type=int, default=PATIENCE_LIMIT)
    parser.add_argument("--neg_samp", type=int, default=NEG_SAMP)
    parser.add_argument("--seeds", type=int, nargs='+', default=[1], help='One or more random seeds')
    parser.add_argument("--outdir", "--out_dir", dest="out_dir", type=str, default='outputs', help='Directory to save checkpoints, weights and logs')
    parser.add_argument("--checkpoint_prefix", type=str, default='sasrec_checkpoint_seed_')
    parser.add_argument("--weights_prefix", type=str, default='sasrec_BEST_weights_')
    args = parser.parse_args()

    # ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    seqs, num_items, item_map, movie_to_genre, genres_num = preprocess(args.data_path, args.genre_path)
    print(f"Total processed movie items in catalog: {num_items}")

    hyperparams = {
        "num_items": num_items,
        "max_len": args.max_len,
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "dropout_rate": args.dropout_rate,
        "learning_rate": args.learning_rate,
        "num_genres": genres_num
    }

    for seed in args.seeds:
        print(f"\n--- Running Training Execution with Seed {seed} ---")
        checkpoint_name = os.path.join(args.out_dir, f"{args.checkpoint_prefix}{seed}.pth")
        weights_name = os.path.join(args.out_dir, f"{args.weights_prefix}{seed}.pth")

        runner = SASRec(
            data=args.data_path,
            genre_path=args.genre_path,
            max_len=args.max_len,
            checkpoint_path=checkpoint_name,
            best_weights_path=weights_name,
            out_dir=args.out_dir,
            random_seed=seed,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience_lim=args.patience_lim
        )
        runner.train(**hyperparams)