# %%
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm, trange

# %%
data_df = pd.read_csv('data/booksummaries.txt', sep='\t', names=['BookID', 'Thing', 'Title', 'Author', 'Publication Date', 'Genre', 'Summary'])
data_df = data_df.drop(columns=['Thing', 'Title', 'Author', 'Publication Date'])
data_df = data_df.dropna(ignore_index=True)
# data_df ['BookID', 'Genre', 'Summary']

# %%
words = []
for s in data_df['Summary']:
    words.append(len(s))
# print(np.median(words))
# print(sum(words) / len(data_df['Summary']))

# %%
def process_genre_list(x):
    genre_json_string = x['Genre']
    genre_list = list(json.loads(genre_json_string).values())
    genre_list = [g.lower() for g in genre_list]
    x['Genre'] = genre_list
    return x

# %%
data_df = data_df.apply(process_genre_list, axis=1)

# %%
genres = data_df['Genre']
c = Counter()
for genre_list in genres:
    c.update(genre_list)
most_common = c.most_common(15)
items, counts = zip(*most_common)
plt.barh(items, counts)
# plt.show()

# %%
genres_to_find = list(items)
count = 0
for x in data_df['Genre']:
    if any(g in x for g in genres_to_find):
        count += 1
# print(f"{count/len(data_df['Genre'])}, {count/len(data_df['Genre']) * len(data_df['Genre'])}")

# %%
genre_counter = {}
for genre in genres_to_find:
    genre_counter[genre] = c[genre]

# %%
def filter_genres(x):
    # filter dataset to only include top 15 most common genres
    x_genres = x['Genre']
    x_genres = [g for g in x_genres if g in genre_counter]
    if len(x_genres) == 0:
        x_genres = np.nan
    x['Genre'] = x_genres
    return x

# %%
filtered_df = data_df.apply(filter_genres, axis=1).dropna().reset_index(drop=True)

# %%
genre_parents = {
    'fiction': [],
    'speculative fiction': ['fiction'],
    'science fiction': ['speculative fiction', 'fiction'],
    'fantasy': ['speculative fiction', 'fiction'],
    'horror': ['speculative fiction', 'fiction'],
    'novel': ['fiction'],
    'historical fiction': ['novel', 'fiction'],
    'romance novel': ['novel', 'fiction'],
    'thriller': ['novel', 'fiction'],
    'suspense': ['novel', 'fiction'],
    'crime fiction': ['novel', 'fiction'],
    'mystery': ['novel', 'fiction'],
    'young adult literature': ['novel', 'fiction'],
    "children's literature": ['novel', 'fiction'],
    'historical novel': ['novel', 'fiction']
}
genre_to_id = {
    'fiction': 0,
    'speculative fiction': 1,
    'science fiction': 2,
    'fantasy': 3,
    'horror': 4,
    'novel': 5,
    'historical fiction': 6,
    'romance novel': 7,
    'thriller': 8,
    'suspense': 9,
    'crime fiction': 10,
    'mystery': 11,
    'young adult literature': 12,
    "children's literature": 13,
    'historical novel': 14
}
id_to_genre = {v: k for k, v in genre_to_id.items()}

# %%
def generate_genre_vector(x):
    genre_list = x['Genre']
    genre_vector = np.zeros(len(genre_to_id))
    for genre in genre_list:
        genre_vector[genre_to_id[genre]] = 1
        for gp in genre_parents[genre]:  # mark parent genres as well
            genre_vector[genre_to_id[gp]] = 1
    x['Label'] = genre_vector
    return x

# %%
labeled_df = filtered_df.apply(generate_genre_vector, axis=1).reset_index(drop=True)

# %% [markdown]
# ## Create Train Test Split 80/20

# %%
train_df, test_df = train_test_split(labeled_df, test_size=0.2, random_state=42)

print("Data splits created")

# %% [markdown]
# ## Important Variables
# 
# genre_parents: genre --> list of parents
# 
# genre_to_id: genre --> id (same order as genre_parents)
# 
# labeled_df: columns = [BookID, Genre, Summary, Label] (Genre is from the original dataset while Label includes parents) 
# 
# train_df: 80% of labeled_df
# 
# test_df: 20% of labeled_df

# %%
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
num_genres = len(genre_to_id)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_genres)

# %%
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, num_genres),
    nn.Sigmoid()  # sigmoid activation for multi label
)

print("Model loaded")

# %%
class SummaryDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = tokenizer(row['Summary'], return_tensors='pt', truncation=True)  # will need to collate
        input_ids = inputs['input_ids'].squeeze()
        genre_label = torch.tensor(row['Label']).float()
        return input_ids, genre_label

def collate_fn(batch):
    input_ids, labels = zip(*batch)
    pad_id = tokenizer.pad_token_id
    padded_input_ids = nn.utils.rnn.pad_sequence(
        sequences=input_ids,
        batch_first=True,
        padding_value=pad_id
    )
    attn_mask = padded_input_ids != pad_id  # 1 when not pad and 0 when pad
    labels = torch.stack([l for l in labels])
    # labels don't need to be padded bc they're all length 15
    return padded_input_ids, attn_mask, labels

# %%
tree_depth = max([len(parents) for parents in genre_parents.values()]) + 1
print(f"tree depth {tree_depth}")

def find_most_specific_genres(genre_vec):
    # returns the names of the most specific genres
    max_depth = 0
    most_specific = []
    for genre_id, _ in enumerate(genre_vec):
        if genre_vec[genre_id] == 0: continue
        depth = len(genre_parents[id_to_genre[genre_id]]) + 1
        if depth == max_depth:
            most_specific.append(id_to_genre[genre_id])
        elif depth > max_depth:
            max_depth = depth
            most_specific = []
            most_specific.append(id_to_genre[genre_id])
    return most_specific

def genre_distance(g1, g2, genre_parents):
    if g1 == g2: 
        return 0
    elif (g1 in genre_parents[g2]) or (g2 in genre_parents[g1]):
        return abs(len(genre_parents[g1])-len(genre_parents[g2]))
    else:
        # dist = d(g1, root) + d(g2, root) - 2*d(lca, root)
        # lca = lowest common ancestor
        lca_dist = 0
        if ("speculative fiction" in genre_parents[g1]) and ("speculative fiction" in genre_parents[g2]):
            lca_dist = 1
        elif ("novel" in genre_parents[g1]) and ("novel" in genre_parents[g2]):
            lca_dist = 1
        return len(genre_parents[g1]) + len(genre_parents[g2]) -2*lca_dist

# %%
class HierarchicalLoss(nn.Module):
    def __init__(self, genre_parents, genre_to_id, id_to_genre, pred_threshold):
        super(HierarchicalLoss, self).__init__()
        self.genre_parents = genre_parents
        self.genre_to_id = genre_to_id
        self.id_to_genre = id_to_genre
        self.pred_threshold = pred_threshold
    
    def forward(self, predictions, targets):
        # predictions, targets shape: [batch_size, num_genres]
        # first compute binary entropy loss per summary in the batch
        base_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')  # shape [batch_size, num_genres]
        per_summary_loss = base_loss.sum(dim=1)  # shape [batch_size, ]
        
        # extra loss is function of distance between prediction and closest most specific genre
        # normalized by some function of tree depth
        for i in range(base_loss.shape[0]):
            most_specific_genres = find_most_specific_genres(targets[i])
            # for all logits over the threshold (predictions), calculate extra loss
            for j in range(len(predictions[i])):
                if predictions[i][j]>=self.pred_threshold:
                    extra_loss = min(
                        [genre_distance(m, self.id_to_genre[j], self.genre_parents) for m in most_specific_genres]
                    )  # both arguments to genre_distance are strs
                    per_summary_loss[i] += extra_loss/np.sqrt(tree_depth)

        per_summary_loss /= len(self.genre_parents)  # divide by num genres to get mean per summary
        return per_summary_loss.mean()

# %% [markdown]
# # Training

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

lr = 1e-4
epochs = 10
batch_size = 32

train_dataloader = DataLoader(
    SummaryDataset(train_df, tokenizer),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
pred_threshold = 0.5
hierarchical_loss = HierarchicalLoss(genre_parents, genre_to_id, id_to_genre, pred_threshold=pred_threshold)

for epoch in trange(epochs, desc="Epoch"):
    model.train()
    for input_ids, attn_mask, labels in tqdm(train_dataloader, desc="inner loop"):
        input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attn_mask)
        loss = hierarchical_loss(outputs.logits, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}: loss {loss.item()}")

# %%
torch.save(model.state_dict(), "final_model")
print("Model Saved")

# %% [markdown]
# # Evaluation

# %%
# Always fiction
print("Always fiction")
y_true = labeled_df['Label']
y_true = np.stack(y_true, axis=0)
y_pred = np.zeros_like(y_true)
y_pred[:,0] = 1
precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
print(precision, recall, fscore)

# %%
# Random
print("Random")
precisions, recalls, fscores = [], [], []
m, n = y_true.shape
for _ in range(50):
    y_pred2 = np.zeros((m,n))
    k = np.random.randint(0, m*n)
    indices = np.random.choice(m*n, k, replace=False)
    y_pred2[np.unravel_index(indices, y_pred2.shape)] = 1
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred2, average='micro')
    precisions.append(precision)
    recalls.append(recall)
    fscores.append(fscore)
print(f"{sum(precisions)/len(precisions)}, {sum(recalls)/len(recalls)}, {sum(fscores)/len(fscores)}")


# %%
test_dataloader = DataLoader(
    SummaryDataset(test_df, tokenizer),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

Y_preds = []
Y_trues = []
with torch.no_grad():
    for input_ids, attn_mask, labels in tqdm(test_dataloader, desc="evaluation"):
        input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attn_mask)
        binary_preds = (outputs > pred_threshold).float()
        
        Y_preds.append(binary_preds.cpu().numpy())
        Y_trues.append(labels.cpu().numpy())
    Y_preds = np.vstack(Y_preds)
    Y_trues = np.vstack(Y_trues)
    precision, recall, f1_score = precision_recall_fscore_support(Y_trues, Y_preds, average='micro')
print(precision, recall, f1_score)

print("DONE")


