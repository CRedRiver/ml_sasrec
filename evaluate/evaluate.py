import torch
from evaluate.metrics import hit_rate_at_k, ndcg_at_k, mrr_at_k

def evaluate_model(model, dataloader, device, k=10, neg_samp=1000, print_res=False):
    model.eval() 
    
    HR_scores = []
    NDCG_scores = []
    MRR_scores = []
    
    with torch.no_grad(): 
        for eval_inputs, eval_targets, eval_genres in dataloader:
            eval_inputs = eval_inputs.to(device, non_blocking=True)
            eval_targets = eval_targets.to(device, non_blocking=True)
            eval_genres = eval_genres.to(device, non_blocking=True)
            
            seq_output = model(eval_inputs, eval_genres)
            logits = torch.matmul(seq_output, model.item_emb.weight.t())
        
            last_logits = logits[:, -1, :] 
            last_targets = eval_targets[:, -1] 
          
            # Create a matrix of uniform probabilities (1.0) for every item in the catalog
            probs = torch.ones_like(last_logits)
            
            # Mask out the padding token (0) so it is never sampled
            probs[:, 0] = 0.0
            
            probs.scatter_(1, eval_inputs, 0.0)
            
            # Mask out the True Target movie
            probs.scatter_(1, last_targets.unsqueeze(1), 0.0)
            
            # Sample `neg_samp` items for every user in the batch SIMULTANEOUSLY.
            negative_samples = torch.multinomial(probs, num_samples=neg_samp, replacement=False)
            
            # Start with every movie at Negative Infinity
            sampled_logits = torch.full_like(last_logits, -float('inf'))
            
            # UNMASK: Reveal the scores for the random negative samples and for 1 True Target
            sampled_logits.scatter_(1, negative_samples, last_logits.gather(1, negative_samples))
            sampled_logits.scatter_(1, last_targets.unsqueeze(1), last_logits.gather(1, last_targets.unsqueeze(1)))

            _, top_k_movie_ids = torch.topk(sampled_logits, k=k, dim=-1)
            
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
        print(f"Evaluation Results (Target vs {neg_samp} Negatives):")
        print(f"Hit Rate @{k}: {final_hr:.4f}")
        print(f"NDCG @{k}:     {final_ndcg:.4f}")
        print(f"MRR @{k}:      {final_mrr:.4f}")
    
    return final_hr, final_ndcg, final_mrr