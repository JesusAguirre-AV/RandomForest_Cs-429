# decisionTree.py
#   In use :
#   - entropy / gini / misclassification impurity
#   - information gain
#   - chi-square prepruning (alpha)


import pandas as pd
import numpy as np
from scipy.stats import chi2


class decisionTree:
    """
    Parameters
    ----------
    D : pd.DataFrame
        full training set (features + target). the target column name is fixed
        to 'isFraud' below (self.target).
    impurity : {'entropy', 'gini', 'misclassification'}
        which impurity function to use at each split.
    alpha : float or None
        if provided, enables chi-square pre-pruning at each candidate split.
        stop splitting if the association between child-branch and class
        is NOT statistically significant at (1 - alpha).
    max_depth : int or None
        maximum depth of the tree. Depth counts edges from root (root depth=0).
    min_samples_split : int
        do not split a node if it has fewer than this many samples.
    min_samples_leaf : int
        after a split, each child must have at least this many samples.
    max_features : int or None
        number of features to consider per split
        if None, use all features.
    numeric_thresholds : int
        Upper bound on candidate numeric thresholds per feature. We generate
        evenly spaced quantile cutpoints to avoid O(U) midpoints (fast!).
    """

    def __init__(self, D, impurity='entropy', alpha=None,
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, numeric_thresholds=32):
        #store training data and hyperparameters
        self.D = D
        self.target = 'isFraud'
        self.impurity_name = impurity
        self.alpha = alpha
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.numeric_thresholds = numeric_thresholds








    #BUILD
    def DT_construction(self):
        """public entry to build the tree from the stored dataset."""
        return self.DT_construct(self.D)

    def DT_construct(self, D, depth=0):
        """
        recursively grow the tree.
        steps:
          1) make a node and set its label to the majority class in D.
          2) check stopping rules (pure, depth limit, too small).
          3) choose best split (feature and possibly threshold).
          4) optionally apply chi-square pre-pruning to reject weak splits.
          5) recurse on child subsets and attach successors.
        """
        #create a new node and assign the majority label (used if we stop here)
        t = self.newNode()
        t.label = self.representativeClass(D)

        #stopping rules
        if (not self.impure(D)) \
                or (self.max_depth is not None and depth >= self.max_depth) \
                or (len(D) < self.min_samples_split):
            #leaf node: return with majority label already set
            return t

        # best splitting criterion:
        #   - categorical: return feature name (str)
        #   - numeric: return tuple (feature_name, threshold)
        criterion = self.splitCriterion(D)
        if criterion is None:
            #no useful split found (e.g., all features constant) -> leaf
            return t

        #partition data by the chosen split
        subsets = self.decompose(D, criterion)

        #min leaf size constraint: reject split if any child too small
        if any(len(s) < self.min_samples_leaf for s in subsets):
            return t

        # Chi-square pre-pruning
        # if alpha is provided and the split is NOT significant,
        # we keep this node as a leaf.
        if self.alpha is not None:
            if not self.chiSquare_significant(D, subsets, self.alpha):
                return t

        #store split info at node for prediction-time routing
        t.criterion = criterion
        if isinstance(criterion, str):
            # for categorical features, we need to remember the order of
            # categories encountered in training to route test rows.
            t.split_values = list(D[criterion].fillna('**MISSING**').unique())

        #Recurse on children
        for s in subsets:
            self.addSuccessor(t, self.DT_construct(s, depth + 1))
        return t



    def DT_classify(self, x, t):
        """
        Predict the class for a single row x, starting from tree node t.
        If t is a leaf, return its label; otherwise route x to the proper child.
        """
        if self.isLeaf(t):
            return t.label
        return self.DT_classify(x, self.splitSuccessor(t, x))








    #SPLITTING

    def splitCriterion(self, dSet):
        """
        search over a set of features and choose the split
        that maximizes information gain for the requested impurity.

        for numeric features:
            - build a small set of candidate thresholds via quantiles.
            - evaluate left/right split for each threshold.

        for categorical features:
            - multi-way split on unique values (including '**MISSING**').

        returns
        -------
        str  -> categorical feature name
        (str, float) -> numeric feature + threshold
        None -> no viable split
        """
        features = dSet.drop(columns=[self.target]).columns.tolist()

        #optional feature subsampling
        if self.max_features is not None and self.max_features < len(features):
            rng = np.random.default_rng()
            features = list(rng.choice(features, size=self.max_features, replace=False))

        best_gain = -1.0
        best_feature = None

        for f in features:
            col = dSet[f]

            if pd.api.types.is_numeric_dtype(col):
                # numeric feature: candidate thresholds via quantiles
                # this avoids scanning all midpoints between unique values
                uq = col.dropna().unique()
                if uq.size <= 1:
                    continue  # no split possible

                # choose at most numeric_thresholds+1 quantiles, then midpoints
                qs = np.linspace(0, 1, num=min(self.numeric_thresholds, uq.size + 1), endpoint=True)
                cand = np.unique(np.quantile(col.dropna(), qs))
                mids = (cand[:-1] + cand[1:]) / 2.0  # candidate thresholds

                for thr in np.unique(mids):
                    left = dSet[col <= thr]
                    right = dSet[col > thr]
                    if len(left) == 0 or len(right) == 0:
                        continue  # skip degenerate split

                    gain = self.informationGain(dSet, [left, right])
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = (f, float(thr))

            else:
                # categorical feature: one branch per unique value
                col_filled = col.fillna('**MISSING**')
                values = col_filled.unique()
                if values.size <= 1:
                    continue  # no split possible

                subsets = [dSet[col_filled == v] for v in values]
                gain = self.informationGain(dSet, subsets)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f

        return best_feature

    def decompose(self, dSet, criterion):
        """
        materialize the child subsets for a split criterion.
        """
        if isinstance(criterion, tuple):
            # numeric split: two children (<= threshold) and (> threshold)
            f, thr = criterion
            col = dSet[f]
            return [dSet[col <= thr], dSet[col > thr]]
        else:
            # categorical split: one child per observed category (including MISSING)
            col = dSet[criterion].fillna('**MISSING**')
            return [dSet[col == v] for v in col.unique()]

    def splitSuccessor(self, node, x):
        """
        route a single row x down from 'node' to the appropriate child.

        handles:
          - numeric splits (<= threshold goes left)
          - categorical splits (match category by the stored order)
          - missing/unseen categories (fallback to first/majority child)
        """
        crit = node.criterion

        # numeric split
        if isinstance(crit, tuple):
            f, thr = crit
            val = x.get(f, np.nan)
            try:
                # send NaNs or non-comparable values to the 'right' child by default
                return node.successors[0] if (pd.notna(val) and val <= thr) else node.successors[1]
            except Exception:
                # Extremely defensive fallback
                return node.successors[0]

        # ---- categorical split ----
        feat = crit
        val = x.get(feat, '**MISSING**')
        if pd.isna(val):
            val = '**MISSING**'
        try:
            idx = node.split_values.index(val)  # which branch matches 'val'?
            return node.successors[idx]
        except Exception:
            # unseen category at test time -> fallback to first child
            return node.successors[0]




    # IMPURITY / GAIN / CHI-SQUARE

    def impure(self, dSet):
        """
        return true if node is 'impure' i.e. has more than one class present
        """
        return dSet[self.target].nunique(dropna=False) >= 2

    def representativeClass(self, dSet):
        """
        majority class label at this node used for:
          - leaf prediction
          - tie-breaks
        """
        return dSet[self.target].mode(dropna=False)[0]

    def _probs(self, dSet):
        """
        class probabilities at this node as a numpy array
        safe when empty.
        """
        counts = dSet[self.target].value_counts(dropna=False)
        total = counts.sum()
        return (counts / total).values if total > 0 else np.array([])

    # ---- impurity functions ----

    def entropy(self, dSet):
        """
        shannon entropy: -sum(p * log2 p)
        """
        p = self._probs(dSet)
        if p.size == 0:
            return 0.0
        return float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    def giniIndex(self, dSet):
        """
        gini impurity: 1 - sum(p^2)
        """
        p = self._probs(dSet)
        return float(1.0 - (p ** 2).sum())

    def misclassificationError(self, dSet):
        """
        misclassification error: 1 - max(p)
        """
        p = self._probs(dSet)
        return float(1.0 - (p.max() if p.size else 1.0))

    def impurity(self, dSet):
        """
        dispatch to the selected impurity function
        defaults to entropy if an unknown name is provided
        """
        if self.impurity_name == 'entropy':
            return self.entropy(dSet)
        if self.impurity_name == 'gini':
            return self.giniIndex(dSet)
        if self.impurity_name == 'misclassification':
            return self.misclassificationError(dSet)
        # fallback (robustness)
        return self.entropy(dSet)

    def informationGain(self, dSet, subsets):
        """
        information Gain = impurity(parent) - sum_i (|child_i|/|parent|) * impurity(child_i)

        works for any impurity choice (entropy/gini/misclassification)
        """
        parent = self.impurity(dSet)
        N = len(dSet)
        weighted = 0.0
        for s in subsets:
            if len(s) == 0:
                continue
            weighted += (len(s) / N) * self.impurity(s)
        return float(parent - weighted)

    def chiSquare_significant(self, parent, subsets, alpha):
        """
        if test is NOT significant at (1 - alpha), we do NOT split (pre-prune).

        steps:
          1) build contingency table O[branches x classes].
          2) compute expected counts under independence: E = (row_sum * col_sum) / total.
          3) compute X^2 = sum((O - E)^2 / E).
          4) compare with chi2.ppf(1 - alpha, df=(B-1)*(K-1)).
        """
        # classes in a fixed order for stable indexing
        cls_vals = parent[self.target].value_counts(dropna=False).index.tolist()
        B = len(subsets)  # number of branches
        K = len(cls_vals)  # number of classes
        if B <= 1 or K <= 1:
            return False  # split is meaningless / cannot test

        # observed counts O
        O = np.zeros((B, K), dtype=float)
        for b, sb in enumerate(subsets):
            vc = sb[self.target].value_counts(dropna=False)
            for k, c in enumerate(cls_vals):
                O[b, k] = float(vc.get(c, 0))

        # row/column totals and grand total
        row_sum = O.sum(axis=1, keepdims=True)
        col_sum = O.sum(axis=0, keepdims=True)
        total = O.sum()
        # if any row/column is all zeros, or total is zero, cannot test
        if total == 0 or (row_sum == 0).any() or (col_sum == 0).any():
            return False

        # expected counts under independence
        E = row_sum @ col_sum / total

        # chi-square statistic
        with np.errstate(divide='ignore', invalid='ignore'):
            stat = np.nansum((O - E) ** 2 / E)

        # degrees of freedom and critical value
        df = (B - 1) * (K - 1)
        crit = chi2.ppf(1 - alpha, df)

        # if stat >= crit -> significant -> allow split
        return bool(stat >= crit)






    # NODES

    def newNode(self):
        """create a fresh node object"""
        return Node()

    def addSuccessor(self, n, s):
        """attach child node s to parent node n"""
        n.successors.append(s)

    def isLeaf(self, n):
        """a node is a leaf if it has no successors"""
        return len(n.successors) == 0


class Node:
    """
    simple container for a decision node or leaf.

    fields
    ------
    successors : list[Node]
        children of this node; empty list means this is a leaf.
    label : any
        majority class label at this node (used at leaves and as fallback).
    criterion : str or (str, float) or None
        split criterion:
          - categorical: feature name (str)
          - numeric: (feature_name, threshold)
          - none: leaf
    split_values : list or None
        for categorical splits, the observed category order during training.
        ssed at prediction time to route rows (Missing -> fallback)
    """
    def __init__(self):
        self.successors = []
        self.label = None
        self.criterion = None       # str for categorical, (feature, threshold) for numeric
        self.split_values = None    # only for categorical
