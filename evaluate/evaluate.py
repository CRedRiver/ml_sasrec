import torch
from evaluate.metrics import hit_rate_at_k, ndcg_at_k, mrr_at_k

def evaluate_model(model, dataloader, device, k=10, print_res=False):
    model.eval() 
    
    HR_scores = []
    NDCG_scores = []
    MRR_scores = []
    
    with torch.no_grad(): 
        for eval_inputs, eval_targets, eval_genres in dataloader:
            eval_inputs = eval_inputs.to(device)
            eval_targets = eval_targets.to(device)
            eval_genres = eval_genres.to(device)
            
            seq_output = model(eval_inputs, eval_genres)
            logits = torch.matmul(seq_output, model.item_emb.weight.t())
            
            # 1. Isolate the very last timestep (The "Leave-One-Out" Target)
            last_logits = logits[:, -1, :] 
            last_targets = eval_targets[:, -1] 
            
            # 2. Mask out the padding token (0)
            last_logits[:, 0] = -float('inf') 
            
            # 3. MASK OUT THE USER'S HISTORY
            # We use scatter_ to look at all the movie IDs in eval_inputs
            # and set their probabilities to Negative Infinity. 
            # This ensures the model CANNOT recommend a movie the user already watched.
            last_logits.scatter_(1, eval_inputs, -float('inf'))
            
            # Extract the top k movies
            _, top_k_movie_ids = torch.topk(last_logits, k=k, dim=-1)
            
            # Convert tensors back to standard Python lists
            recommended_lists = top_k_movie_ids.cpu().numpy().tolist()
            target_lists = last_targets.cpu().numpy().tolist()
            
            for recommended, target in zip(recommended_lists, target_lists):
                if target == 0: 
                    continue
                    
                ground_truth = [target]
                
                HR_scores.append(hit_rate_at_k(recommended, ground_truth, k))
                NDCG_scores.append(ndcg_at_k(recommended, ground_truth, k))
                MRR_scores.append(mrr_at_k(recommended, ground_truth, k))
                
    final_hr = sum(HR_scores) / len(HR_scores) if HR_scores else 0
    final_ndcg = sum(NDCG_scores) / len(NDCG_scores) if NDCG_scores else 0
    final_mrr = sum(MRR_scores) / len(MRR_scores) if MRR_scores else 0
    
    if print_res:
        print(f"Evaluation Results:")
        print(f"Hit Rate @{k}: {final_hr:.4f}")
        print(f"NDCG @{k}:     {final_ndcg:.4f}")
        print(f"MRR @{k}:      {final_mrr:.4f}")
    
    return final_hr, final_ndcg, final_mrr