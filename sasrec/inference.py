import os
import argparse
import torch
from sasrec.preprocess import preprocess
from sasrec.sasrecnet import SASRecNet


class SASRecInference:
    def __init__(self, model, device, item_map, movie_to_genre=None, weights="sasrec_BEST_weights_1.pth"):
        self.device = device
        self.item_map = item_map or {}
        self.reverse_map = {internal_id: raw_id for raw_id, internal_id in self.item_map.items()}
        self.movie_to_genre = movie_to_genre or {}

        # load checkpoint/state dict 
        try:
            ckpt = torch.load(weights, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from {weights}: {e}")

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt

        # strip "module." prefix if present (DataParallel artifacts)
        new_state = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            new_state[new_k] = v

        model.load_state_dict(new_state)
        self.model = model.to(self.device)
        self.model.eval()
        self.max_len = getattr(self.model, 'max_len', None)

    def _map_input_ids(self, seq):
        mapped = []
        for x in seq:
            if x == 0:
                mapped.append(0)
            elif x in self.item_map:
                # raw movieId -> internal id
                mapped.append(self.item_map[x])
            elif x in self.reverse_map:
                # already an internal id
                mapped.append(x)
            else:
                # unknown -> padding
                mapped.append(0)
        return mapped

    def recommend(self, user_seqs, k=10, filter_history=True):
        self.model.eval()
        all_recommendations = []

        with torch.no_grad():
            for seq in user_seqs:
                mapped_seq = self._map_input_ids(seq)

                if self.max_len is None:
                    raise RuntimeError("Model max_len not set; cannot create input windows")

                if len(mapped_seq) > self.max_len:
                    processed_seq = mapped_seq[-self.max_len:]
                else:
                    processed_seq = mapped_seq

                pad_len = self.max_len - len(processed_seq)
                padded_seq = ([0] * pad_len) + processed_seq

                item_tensor = torch.tensor([padded_seq], dtype=torch.long, device=self.device)

                if self.movie_to_genre:
                    genre_seq = [self.movie_to_genre.get(i, 0) for i in padded_seq]
                else:
                    genre_seq = [0] * self.max_len

                genre_tensor = torch.tensor([genre_seq], dtype=torch.long, device=self.device)

                seq_output = self.model(item_tensor, genre_tensor)
                logits = torch.matmul(seq_output, self.model.item_emb.weight.t())
                last_logits = logits[:, -1, :].clone()

                # mask padding token
                last_logits[:, 0] = -float('inf')

                # mask items in user history
                if filter_history:
                    for internal_id in set(processed_seq):
                        if internal_id != 0:
                            # guard against out-of-range indices
                            if internal_id < last_logits.size(-1):
                                last_logits[:, internal_id] = -float('inf')

                _, top_k_internal_ids = torch.topk(last_logits, k=k, dim=-1)
                recommended_internal = top_k_internal_ids[0].cpu().numpy().tolist()

                # map back to raw movie IDs when possible
                recommended_raw = [self.reverse_map.get(idx, idx) for idx in recommended_internal]

                all_recommendations.append(recommended_raw)
                print(f"Recommend (MovieLens IDs): {recommended_raw}")

        return all_recommendations


if __name__ == "__main__":
    default_data = r"D:\HUST\2025.2\ML\Project\data\ratings.csv"
    default_genre = r"D:\HUST\2025.2\ML\Project\data\movies.csv"
    default_weights = os.path.join('.', 'sasrec_BEST_weights_1.pth')

    parser = argparse.ArgumentParser(description="SASRec inference CLI")
    parser.add_argument('--data_path', type=str, default=default_data)
    parser.add_argument('--genre_path', type=str, default=default_genre)
    parser.add_argument('--weights', type=str, default=default_weights, help='Path to model weights (.pth)')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--k', type=int, default=5, help='Number of items to recommend')
    parser.add_argument('--no_filter', action='store_true', help='Do not filter items from user history')
    parser.add_argument('--seq', action='append', help='Comma-separated sequence of movie IDs (raw or internal). Use multiple --seq to provide multiple users')
    parser.add_argument('--seq_file', type=str, help='Path to newline-delimited file of comma-separated sequences')
    parser.add_argument('--max_len', type=int, default=400)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    args = parser.parse_args()

    seqs, num_items, item_map, movie_to_genre, num_genres = preprocess(args.data_path, args.genre_path)
    device = torch.device(args.device)

    model = SASRecNet(
        num_items=num_items,
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        num_genres=num_genres,
    )

    inference = SASRecInference(model, device, item_map, movie_to_genre=movie_to_genre, weights=args.weights)

    user_seqs = []
    # load sequences from CLI
    if args.seq_file:
        if os.path.exists(args.seq_file):
            with open(args.seq_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(',') if p.strip()]
                    user_seqs.append([int(x) for x in parts])
        else:
            raise FileNotFoundError(f"seq_file not found: {args.seq_file}")

    if args.seq:
        for s in args.seq:
            parts = [p.strip() for p in s.split(',') if p.strip()]
            user_seqs.append([int(x) for x in parts])

    if not user_seqs:
        # fallback to a few sequences from dataset
        user_seqs = seqs[:3]

    recommendations = inference.recommend(user_seqs, k=args.k, filter_history=(not args.no_filter))
    print("Final recommendations:")
    for i, rec in enumerate(recommendations):
        print(f"User {i}: {rec}")