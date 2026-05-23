import torch
from torch.utils.data import Dataset, DataLoader

class SASRecDataset(Dataset):
    def __init__(self, sequences, movie_to_genre_id, max_len=50, is_training=False, stride=None):
        super().__init__()
        self.max_len = max_len
        self.movie_to_genre_id = movie_to_genre_id
        self.augmented_sequences = []
    
        # Default to max_len (no overlap). 
        if stride is None:
            self.stride = max_len
        else:
            self.stride = stride

        for seq in sequences:
            if len(seq) < 2:
                continue
                
            window_size = self.max_len + 1

            if is_training and len(seq) > window_size:
                end_idx = len(seq)
                
                while end_idx > 1:
                    start_idx = max(0, end_idx - window_size)
                    
                    sub_seq = seq[start_idx:end_idx]
                    self.augmented_sequences.append(sub_seq)
                    
                    end_idx -= self.stride
                    
                    if start_idx == 0:
                        break
            else:
                if len(seq) > window_size:
                    self.augmented_sequences.append(seq[-window_size:])
                else:
                    self.augmented_sequences.append(seq)

    def __len__(self):
        return len(self.augmented_sequences)

    def __getitem__(self, index):
        seq = self.augmented_sequences[index]

        inputs = seq[:-1]
        targets = seq[1:]

        pad_len = self.max_len - len(inputs)
        
        input_seq = ([0] * pad_len) + inputs
        target_seq = ([0] * pad_len) + targets

        genre_seq = ([0] * pad_len) + [self.movie_to_genre_id.get(item, 0) for item in inputs]

        return (torch.tensor(input_seq, dtype=torch.long), 
                torch.tensor(target_seq, dtype=torch.long),
                torch.tensor(genre_seq, dtype=torch.long))