# cython: language_level=3



cdef class TreeNode:
    cdef public int feature
    cdef public double threshold
    cdef public object left, right, value

    def __init__(self):
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = None

cdef class TreeBase:
    cdef public int max_depth, min_samples_split
    cdef public TreeNode root

    def __init__(self, int max_depth=10, int min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    cpdef fit(self, list X, list y): 
        self.root = self._build_tree(X, y, depth=0)

    cpdef predict(self, list X):
        cdef list preds = []
        for x in X:
            preds.append(self._traverse(x self.root))
        return preds
    
    cdef object _traverse(self, list x, TreeNode node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
        
    cdef TreeNode _build_tree(self, list X, list y, int depth):
        cdef int n = len(y)
        node = TreeNode()
        if depth >= self.max_depth or n < self.min_samples_split or self._is_pure(y):
            node.value = self._leaf_value(y)
            return node
        
        feat, thr = self._best_split(X, y)
        if feat < 0:
            node.value = self._leaf_value(y)
            return node

        node.feature = feat
        node.threshold = thr

        Xl, yl, Xr, yr = [], [], [], []
        for xi, yi in zip(X, y):
            if xi[feat] <= thr:
                Xl.append(xi)
                yl.append(yi)
            else:
                Xr.append(xi)
                yr.append(yi)
            
        node.left = self._build_tree(Xl, yl, depth + 1)
        node.right = self._build_tree(Xr, yr, depth + 1)
        return node

    cdef bint _is_pure(self, list y):
        cdef object first = y[0]
        for v in y:
            if v != first:
                return False
        return True

    cdef object _leaf_value(self, list y):
        raise NotImplementedError()
    
    cdef tuple _best_split(self, list X, list y):
        raise NotImplementedError()