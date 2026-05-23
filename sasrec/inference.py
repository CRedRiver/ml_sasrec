import torch
from sasrec.preprocess import preprocess
from sasrec.sasrecnet import SASRecNet

class SASRecInference():
    def __init__(self, model, device, item_map, weights = "sasrec_BEST_weights.pth"):
        state_dict = torch.load(weights, weights_only = True)
        model.load_state_dict(state_dict)
        self.model = model
        self.device = device
        self.max_len = model.max_len
        self.reverse_map = {internal_id: ml_id for ml_id, internal_id in item_map.items()}

    def recommend(self, user_seqs, k=10, filter_history=True):
        self.model.eval()
        all_recommendations = []

        with torch.no_grad():
            for seq in user_seqs:
                if len(seq) > self.max_len:
                    processed_seq = seq[-self.max_len:]
                else:
                    processed_seq = seq
                    
                pad_len = self.max_len - len(processed_seq)
                processed_seq = ([0] * pad_len) + processed_seq
                input_seq = torch.tensor([processed_seq], dtype=torch.long).to(self.device)

                seq_output = self.model(input_seq)
                logits = torch.matmul(seq_output, self.model.item_emb.weight.t())

                last_logits = logits[:, -1, :]
                last_logits[:, 0] = -float('inf')

                if filter_history:
                    for internal_id in seq:
                        if internal_id != 0: 
                            last_logits[:, internal_id] = -float('inf')

                _, top_k_internal_ids = torch.topk(last_logits, k=k, dim=-1)
                recommended_internal = top_k_internal_ids[0].cpu().numpy().tolist()
                
                real_movie_ids = [self.reverse_map.get(idx, idx) for idx in recommended_internal]
                
                all_recommendations.append(real_movie_ids)
                print(f"Recommend (MovieLens IDs): {real_movie_ids}")
                
        return all_recommendations

if __name__ == "__main__":
    seqs, num_items, item_map = preprocess(r"D:\HUST\2025.2\ML\Project\data\ratings.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SASRecNet(num_items=num_items, 
                      max_len=400,
                      hidden_size=32,
                      num_heads=2,
                      dropout_rate = 0.3)
    inference = SASRecInference(model, device, item_map)
    dummy_seqs = [[4,5,2,6],[4,6,3,1,8,9],[4,6,3,1,8,9,5]]
    inference.recommend(dummy_seqs,k=5)