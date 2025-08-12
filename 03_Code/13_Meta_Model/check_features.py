#!/usr/bin/env python3

import numpy as np

# Load validation data
X_val = np.load('../../04_Models/meta/daily/GDEA/X_val.npy')

print(f'Validation shape: {X_val.shape}')
print(f'\nFirst 3 samples (first 5 features):')
print(X_val[:3, :5])

print(f'\nFeature statistics:')
for i in range(min(X_val.shape[1], 15)):
    col = X_val[:, i]
    print(f'  Feature {i:2d}: mean={col.mean():7.4f}, std={col.std():7.4f}, non-zero={(col != 0).mean():.1%}')

print(f'\nOverall non-zero ratio: {(X_val != 0).mean():.1%}')
print(f'Samples with >50% features: {((X_val != 0).mean(axis=1) > 0.5).sum() / len(X_val):.1%}')