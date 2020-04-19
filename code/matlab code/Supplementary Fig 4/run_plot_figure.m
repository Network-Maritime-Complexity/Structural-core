% load data
load('trade_GLSN.mat', 'x')
load('trade_GLSN_metadata.mat', 'module')

% embedding
coords = coalescent_embedding(x, 'RA2', 'ISO', 'EA', 2);

% angular separation
[index, ~, pvalue] = compute_angular_separation(coords(:,1), module, 1, 10000);