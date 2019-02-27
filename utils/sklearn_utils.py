from sklearn.pipeline import FeatureUnion, _fit_one_transformer, _transform_one
from sklearn.utils._joblib import delayed, Parallel

class SegmentedFeatureUnion(FeatureUnion):
    """
    Applies transformers to selections of features.
    Class based on sklearn.pipeline FeatureUnion
    """
    
    def __init__(self, transformer_list, n_jobs = None,
                 transformer_weights = None, transformer_ranges = None):
        """
        Parameters:
            transformer_list ([object]) : list of objects that implements the methods 
                fit : fit(X[, y])
                transform : transform(X, [,y])
            transformer_ranges ([np.ndarray]) : list of indices to which the transformers are applied. 
        """
        
        super().__init__(transformer_list, 
                         n_jobs = n_jobs, 
                         transformer_weights = transformer_weights)
        
        self.transformer_ranges = transformer_ranges

    def _iter(self):
        """
        Generate (name, trans, weight, transformer_range) tuples excluding None and
        'drop' transformers.
        """
        get_weight = (self.transformer_weights or {}).get
        
        return ((name, trans, get_weight(name), transformer_range)
                    for (name, trans), transformer_range in zip(self.transformer_list, self.transformer_ranges)
                        if trans is not None and trans != 'drop')
    
    def fit(self, X, y = None):
        """
        Fit all transformers using X.
        
        Parameters:
            X (np.ndarray[n_samples, n_features]) : data to be transformed
            y ({np.ndarray[n_samples], None}) : target variable, usually not used
            
        Returns:
            self (SegmentedFeatureUnion) : this estimator
        """
        
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_transformer)(trans, X[:, transform_range], y)
                for _, trans, _, transform_range in self._iter())
        
        # save fitted transformers --> used in self._iter
        self._update_transformer_list(transformers)
        
        return self

    def transform(self, X, y = None):
        """
        Transform X separately by each transformer, concatenate results.
        Parameters
            X (np.ndarray[n_samples, n_features]) : data to be transformed
            y ({np.ndarray[n_samples], None}) : ignored
            
        Returns
            X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        
        x_t = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X[:, transform_range], None, weight)
                for name, trans, weight, transform_range in self._iter())
        
        # stack arrays
        X_t = np.hstack(X_t)
        
        return X_t
    
    def fit_transform(self, X, y = None):
        """
        Override parents fittransform.
        """
        return self.fit(X, y).transform(X, y)


def create_flat_transformer(transformer):
    """
    Wrapper to perform the transformation on the flattened input array to 1D
    The X argument is transformmed:
        (n_samples, n_features) --> (n_samples * n_features, 1)
        
    It preserves the methods and attributes of the underlying estimator.
    Parameters:
        transformer (object) : should implement the methods
            fit(X[, y])
            transform(X)
            
    Returns
        inner (class) : class with decorated methods.
    """
    
    class inner(transformer):
        def __init__(self, **kwargs):
            """
            Instantiates a transformer object.
            """
             
            # sanity check 
            required_attrs = ['fit', 'transform', 'fit_transform']
            for attr in required_attrs:
                if getattr(super(), attr, None) is None:
                    raise AttributeError("transformer should have {0} attribute".format(attr))
             
            # use underlying estimators machinery
            super().__init__( **kwargs)

        def fit(self, X, y = None):
            """
            1. flattens a 2D array
            2. creates rule for transform on the flattened array.
            """
            
            super().fit(X.reshape(-1, 1), y)
            
            return self
   
        def transform(self, X):
            """
            1. flattens a 2D array
            2. transforms it
            3. restores it ot its original shape
            """
            
            X_t = super().transform(X.reshape(-1, 1))
            return X_t.reshape(-1, X.shape[1])
        
        def fit_transform(self, X, y = None):
            """
            Fits and transforms 2D array via flattening.
            """
            
            x_t = super().fit_transform(X.reshape(-1, 1), y)
            return x_t.reshape(-1, X.shape[1])

    return inner


class Encoder:
    """
    Lightweight encoder -- decoder class.
    
    Methods:
        encode(x) : returns the code of x
        decode(x) : returns the value that is encoded by x
        reset() : clears memory
    """
    
    def __init__(self):
        
        self._idx = -1
        self._encode_map = {}
        self._decode_map = {}
        
    def encode(self, x):
        """
        Encodes a hashable object with an integer.
        Parameters:
            x (object) : value to encode
            
        Returns:
            (int) code of x
        """
        
        if not x in self._encode_map:
            self._idx += 1
            
            self._encode_map.update({x : self._idx})
            self._decode_map.update({self._idx : x})
            
        return self._encode_map[x]
    
    def decode(self, x):
        """
        Dencodes a hashable object with an integer.
        Parameters:
            x (int) : value to encode
            
        Returns:
            the value that is encoded by x
        """
        return self._decode_map[x]
    
    def reset(self):
        """
        Clears lookup tables.
        """
        
        self._idx = -1
        self._encode_map = {}
        self._decode_map = {}