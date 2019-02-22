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