
import math
import random
from collections import defaultdict



RISK_MULTIPLIERS = {
    'High':   1.5,
    'Medium': 1.2,
    'Low':    1.0,
}

#location type numeric scores for feature extraction
LOCATION_TYPE_SCORES = {
    'Residential':    0.6,
    'School':         0.3,
    'Hospital':       0.2,
    'AmbulanceDepot': 0.2,
    'PowerPlant':     0.7,
    'Industrial':     0.9,
    'Empty':          0.0,
}

KMEANS_MAX_ITER  = 100
KMEANS_RESTARTS  = 5     #number of random restarts to avoid gandi initialisations
KMEANS_K_RANGE   = (2, 5)  #elbow method tests k from 2 to 5


#sabse pehle features ko extract kiya hai 
#which are population density, industries se kitna door, location type ka bhi score hai, and nearby roads blocks ka scene
def _extract_features(graph, grid_size):


    #assign population density if not already set
    for pos, node in graph.nodes.items():
        if node.location_type == 'Empty':
            node.population_density = 0.0
        elif node.population_density == 0.0:

            #assign based on type
            base = {
                'Residential':    random.uniform(0.5, 1.0),
                'Hospital':       random.uniform(0.3, 0.6),
                'School':         random.uniform(0.4, 0.7),
                'Industrial':     random.uniform(0.1, 0.4),
                'PowerPlant':     random.uniform(0.1, 0.3),
                'AmbulanceDepot': random.uniform(0.2, 0.5),
            }
            node.population_density = base.get(node.location_type, 0.0)


    #collect industrial positions for proximity calculation
    industrial_positions = [
        pos for pos, node in graph.nodes.items()
        if node.location_type == 'Industrial'
    ]

    def industrial_proximity(pos):
        if not industrial_positions:
            return 0.0
        r, c = pos
        min_dist = min(
            abs(r - ir) + abs(c - ic)
            for ir, ic in industrial_positions
        )
        #normalising horhi and max possible manhattan distance on grid is below
        max_dist = 2 * (grid_size - 1)
        #closer = higher score
        return 1.0 - (min_dist / max_dist)

    #scale population density into [0, 1] based on the city distribution
    density_values = [
        node.population_density for _, node in graph.nodes.items()
        if node.location_type != 'Empty'
    ]
    if density_values:
        d_min = min(density_values)
        d_max = max(density_values)
    else:
        d_min, d_max = 0.0, 0.0

    def scale_density(d):
        span = d_max - d_min
        return (d - d_min) / span if span > 0 else 0.0

    def nearby_blockage_density(pos, radius_hops=2):

        r0, c0 = pos
        blocked_edges = 0
        total_edges = 0

        for dr in range(-radius_hops, radius_hops + 1):
            for dc in range(-radius_hops, radius_hops + 1):
                if abs(dr) + abs(dc) > radius_hops:
                    continue
                u = (r0 + dr, c0 + dc)
                if u not in graph.nodes:
                    continue
                ur, uc = u
                for vr, vc in ((ur - 1, uc), (ur + 1, uc), (ur, uc - 1), (ur, uc + 1)):
                    v = (vr, vc)
                    if v not in graph.nodes:
                        continue
                    #counting each undirected edge once.
                    if u > v:
                        continue
                    total_edges += 1
                    if (u, v) in graph.blocked or (v, u) in graph.blocked:
                        blocked_edges += 1

        return (blocked_edges / total_edges) if total_edges else 0.0

    positions = []
    feature_matrix = []

    for pos, node in graph.nodes.items():
        if node.location_type == 'Empty':
            continue
        f0 = scale_density(node.population_density)
        f1 = industrial_proximity(pos)
        f2 = LOCATION_TYPE_SCORES.get(node.location_type, 0.0)
        f3 = nearby_blockage_density(pos, radius_hops=2)
        positions.append(pos)
        feature_matrix.append([f0, f1, f2, f3])

    return positions, feature_matrix

#minmax normalising each feature column to [0, 1]
def _normalise(feature_matrix):

    if not feature_matrix:
        return feature_matrix
    n_features = len(feature_matrix[0])
    mins = [min(row[i] for row in feature_matrix) for i in range(n_features)]
    maxs = [max(row[i] for row in feature_matrix) for i in range(n_features)]
    normalised = []
    for row in feature_matrix:
        norm_row = []
        for i in range(n_features):
            span = maxs[i] - mins[i]
            norm_row.append((row[i] - mins[i]) / span if span > 0 else 0.0)
        normalised.append(norm_row)
    return normalised


#YAHAN K MEANS HORHI CLUSTERING
def _euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _kmeans_once(data, k):

    # initialise centroids by picking k random distinct points
    centroids = random.sample(data, k)
    centroids = [list(c) for c in centroids]
    labels    = [0] * len(data)

    for _ in range(KMEANS_MAX_ITER):
        #assignment step
        new_labels = []
        for point in data:
            dists     = [_euclidean(point, c) for c in centroids]
            new_labels.append(dists.index(min(dists)))

        #check convergence
        if new_labels == labels:
            break
        labels = new_labels

        #update step
        new_centroids = []
        for ki in range(k):
            cluster_points = [data[i] for i, l in enumerate(labels) if l == ki]
            if not cluster_points:
                #empty cluster so re-initialise to a random point
                new_centroids.append(list(random.choice(data)))
            else:
                n_feat = len(cluster_points[0])
                mean   = [sum(p[f] for p in cluster_points) / len(cluster_points)
                          for f in range(n_feat)]
                new_centroids.append(mean)
        centroids = new_centroids


    inertia = sum(
        _euclidean(data[i], centroids[labels[i]]) ** 2
        for i in range(len(data))
    )
    return labels, centroids, inertia


def _kmeans(data, k):
    #kMeans with multiple restart
    best_labels    = None
    best_centroids = None
    best_inertia   = float('inf')

    for _ in range(KMEANS_RESTARTS):
        labels, centroids, inertia = _kmeans_once(data, k)
        if inertia < best_inertia:
            best_inertia   = inertia
            best_labels    = labels
            best_centroids = centroids

    return best_labels, best_centroids, best_inertia


def _elbow_method(data):

    k_min, k_max = KMEANS_K_RANGE
    k_max = min(k_max, len(data) - 1)  #cant have more clusters than points

    inertias = {}
    for k in range(k_min, k_max + 1):
        _, _, inertia = _kmeans(data, k)
        inertias[k] = inertia

    # find elbow
    best_k    = k_min
    best_drop = -1
    ks        = sorted(inertias.keys())
    for i in range(1, len(ks) - 1):
        prev = inertias[ks[i-1]]
        curr = inertias[ks[i]]
        nxt  = inertias[ks[i+1]]
       
        drop = (prev - curr) - (curr - nxt)
        if drop > best_drop:
            best_drop = drop
            best_k    = ks[i]

    return best_k, inertias


#SYNTHETIC CRIME DATA YAHAN GENERATE HORA
def _generate_crime_scores(feature_matrix, cluster_labels):
    """
    crime_score = 0.45 * population_density+ 0.30 * industrial_proximity+ 0.15 * location_type_score+ 0.10 * nearby_blockage_density+ cluster_noise (small random noise per cluster)
    as areas with high population density
    near industrial zones have higher crime likelihood
    """
    #assign a small noise factor per cluster consistent within cluster
    cluster_noise = defaultdict(lambda: random.uniform(-0.08, 0.08))

    scores = []
    for i, features in enumerate(feature_matrix):
        if len(features) >= 4:
            f0, f1, f2, f3 = features
        else:
            f0, f1, f2 = features[:3]
            f3 = 0.0
        cluster    = cluster_labels[i]
        score      = (0.45 * f0) + (0.30 * f1) + (0.15 * f2) + (0.10 * f3) + cluster_noise[cluster]
        scores.append(max(0.0, min(1.0, score)))  #to [0, 1]

    return scores


#top 33 percen high middle 33 percent medium bottom 33 percent low
def _scores_to_labels(scores):

    sorted_scores = sorted(scores)
    n             = len(sorted_scores)
    low_thresh    = sorted_scores[n // 3]
    high_thresh   = sorted_scores[2 * n // 3]

    labels = []
    for s in scores:
        if s >= high_thresh:
            labels.append('High')
        elif s >= low_thresh:
            labels.append('Medium')
        else:
            labels.append('Low')
    return labels


#DECISION TREE CLASSIFICATION YAHAN HORI HAI
def _gini(labels):

    if not labels:
        return 0.0
    counts = defaultdict(int)
    for l in labels:
        counts[l] += 1
    total = len(labels)
    return 1.0 - sum((v / total) ** 2 for v in counts.values())


def _best_split(X, y):

    best_gain      = -1
    best_feat      = None
    best_threshold = None
    n              = len(y)
    parent_gini    = _gini(y)
    n_features     = len(X[0])

    for feat in range(n_features):
        #try midpoint between every consecutive pair of sorted values
        values = sorted(set(row[feat] for row in X))
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i+1]) / 2
            left_y    = [y[j] for j in range(n) if X[j][feat] <= threshold]
            right_y   = [y[j] for j in range(n) if X[j][feat] >  threshold]
            if not left_y or not right_y:
                continue
            weighted_gini = (len(left_y) / n) * _gini(left_y) + \
                            (len(right_y) / n) * _gini(right_y)
            gain = parent_gini - weighted_gini
            if gain > best_gain:
                best_gain      = gain
                best_feat      = feat
                best_threshold = threshold

    return best_feat, best_threshold


class _DecisionTreeNode:
    def __init__(self):
        self.feat      = None      #feature index to split on
        self.threshold = None      #split threshold
        self.left      = None      #left child
        self.right     = None      #right child
        self.label     = None      #set only on leaf nodes


#below is decision tree classifier that splits on gini impurity
class DecisionTree:

    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.root        = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        node = _DecisionTreeNode()

        #leaf conditions
        if len(set(y)) == 1 or len(y) < self.min_samples or depth >= self.max_depth:
            counts    = defaultdict(int)
            for l in y:
                counts[l] += 1
            node.label = max(counts, key=counts.get)
            return node

        feat, threshold = _best_split(X, y)
        if feat is None:
            counts = defaultdict(int)
            for l in y:
                counts[l] += 1
            node.label = max(counts, key=counts.get)
            return node

        node.feat      = feat
        node.threshold = threshold

        left_idx  = [i for i in range(len(y)) if X[i][feat] <= threshold]
        right_idx = [i for i in range(len(y)) if X[i][feat] >  threshold]

        node.left  = self._build([X[i] for i in left_idx],
                                  [y[i] for i in left_idx], depth + 1)
        node.right = self._build([X[i] for i in right_idx],
                                  [y[i] for i in right_idx], depth + 1)
        return node

    def predict_one(self, x):
        node = self.root
        while node.label is None:
            if x[node.feat] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    def predict(self, X):
        return [self.predict_one(x) for x in X]


def _train_test_split(X, y, test_ratio=0.2):
    
    combined = list(zip(X, y))
    random.shuffle(combined)
    split    = int(len(combined) * (1 - test_ratio))
    train    = combined[:split]
    test     = combined[split:]
    X_train  = [item[0] for item in train]
    y_train  = [item[1] for item in train]
    X_test   = [item[0] for item in test]
    y_test   = [item[1] for item in test]
    return X_train, y_train, X_test, y_test


def _accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)












def run_challenge5(graph, grid_size):


    logs = []

    #features extract
    positions, features = _extract_features(graph, grid_size)

    if len(positions) < 4:
        logs.append("C5: Not enough non-Empty nodes for clustering.")
        return {}, 0.0, logs, {}

    logs.append(f"C5: Extracted features from {len(positions)} nodes.")
    

    features = _normalise(features)

    #kmeans
    logs.append("C5: Running K-Means clustering (elbow method)...")
    optimal_k, inertias = _elbow_method(features)
    logs.append(f"C5: Elbow method selected k={optimal_k}.")

    cluster_labels, centroids, _ = _kmeans(features, optimal_k)

    cluster_counts = defaultdict(int)
    for l in cluster_labels:
        cluster_counts[l] += 1
    for ki, count in sorted(cluster_counts.items()):
        logs.append(f"  Cluster {ki}: {count} nodes")

    #crime data
    logs.append("C5: Generating synthetic crime labels...")
    crime_scores  = _generate_crime_scores(features, cluster_labels)
    crime_labels  = _scores_to_labels(crime_scores)

    high_count   = crime_labels.count('High')
    medium_count = crime_labels.count('Medium')
    low_count    = crime_labels.count('Low')
    logs.append(f"C5: Labels — High: {high_count}, Medium: {medium_count}, Low: {low_count}")

    #decison tree
    logs.append("C5: Training Decision Tree classifier...")

    X_train, y_train, X_test, y_test = _train_test_split(features, crime_labels, test_ratio=0.25)

    tree = DecisionTree(max_depth=5, min_samples=2)
    tree.fit(X_train, y_train)

    #evaluate on test set
    y_pred   = tree.predict(X_test)
    accuracy = _accuracy(y_test, y_pred)
    logs.append(f"C5: Decision Tree trained. Test accuracy: {accuracy*100:.1f}%")

    #predict for ALL nodes for final risk assignment
    all_predictions = tree.predict(features)

    #risk indexes city graph mein likhdiye
    logs.append("C5: Writing risk indices to shared city graph...")
    risk_map = {}
    for i, pos in enumerate(positions):
        risk_label = all_predictions[i]
        multiplier = RISK_MULTIPLIERS[risk_label]
        graph.nodes[pos].risk_index = multiplier
        risk_map[pos] = risk_label

    
    logs.append(
        f"C5: Complete. Risk multipliers applied — "
        f"High(x1.5): {list(all_predictions).count('High')}, "
        f"Medium(x1.2): {list(all_predictions).count('Medium')}, "
        f"Low(x1.0): {list(all_predictions).count('Low')}."
    )
    logs.append("C5: Edge costs updated. A* and GA will now use risk-weighted paths.")

    stats = {
        'k':            optimal_k,
        'n_nodes':      len(positions),
        'cluster_counts': dict(cluster_counts),
        'accuracy':     accuracy,
        'high':         list(all_predictions).count('High'),
        'medium':       list(all_predictions).count('Medium'),
        'low':          list(all_predictions).count('Low'),
    }

    return risk_map, accuracy, logs, stats