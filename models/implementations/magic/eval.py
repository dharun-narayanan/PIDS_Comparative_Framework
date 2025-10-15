"""
Evaluation utilities for MAGIC model.
"""

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def evaluate_entity_level_using_knn(embeddings, labels, train_ratio=0.8, k=5):
    """
    Evaluate embeddings using K-NN classifier.
    
    Args:
        embeddings: Node embeddings (numpy array or torch tensor)
        labels: Node labels
        train_ratio: Ratio of training data
        k: Number of neighbors for KNN
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Shuffle and split
    indices = np.random.permutation(len(embeddings))
    train_size = int(len(embeddings) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings[train_indices], labels[train_indices])
    
    # Predict
    predictions = knn.predict(embeddings[test_indices])
    pred_proba = knn.predict_proba(embeddings[test_indices])
    
    # Compute metrics
    test_labels = labels[test_indices]
    
    results = {
        'f1': f1_score(test_labels, predictions, average='binary' if len(np.unique(labels)) == 2 else 'macro'),
        'precision': precision_score(test_labels, predictions, average='binary' if len(np.unique(labels)) == 2 else 'macro', zero_division=0),
        'recall': recall_score(test_labels, predictions, average='binary' if len(np.unique(labels)) == 2 else 'macro', zero_division=0),
    }
    
    # Compute AUC if binary classification
    if len(np.unique(labels)) == 2 and pred_proba.shape[1] == 2:
        try:
            results['auc'] = roc_auc_score(test_labels, pred_proba[:, 1])
        except:
            results['auc'] = 0.0
    else:
        results['auc'] = 0.0
    
    return results


def batch_level_evaluation(model, dataloader, device):
    """
    Perform batch-level evaluation.
    
    Args:
        model: MAGIC model
        dataloader: Data loader
        device: Device to run on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            
            # Get embeddings
            embeddings = model.embed(batch)
            all_embeddings.append(embeddings.cpu())
            
            # Get labels if available
            if hasattr(batch, 'y'):
                all_labels.append(batch.y.cpu())
            elif hasattr(batch, 'label'):
                all_labels.append(batch.label.cpu())
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    if len(all_labels) > 0:
        all_labels = torch.cat(all_labels, dim=0)
        # Evaluate using KNN
        results = evaluate_entity_level_using_knn(all_embeddings, all_labels)
    else:
        # No labels available, return embeddings only
        results = {'embeddings': all_embeddings}
    
    return results
