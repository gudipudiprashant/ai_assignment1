class SearchAlgo:
  """
  Base class for Search Algorithms for feature selection.
  """
  def __init__(self, all_features, obj_fn):
    """
    Args:
      all_features  - list of strings denoting the features, i.e, feature names
                      of the data to be optimized/selected
      obj_fn        - function object to be called with a single parameter - 
                      an unordered list of strings denoting the features, that
                      evaluates the performance of the given strings.
    """
    self.all_features = all_features
    self.obj_fn = obj_fn

  def encodeFeatures(self, feature_list):
    """
    Returns an encoded string corresponding to the features in feature_list.
    Args:
      feature_list  - list of strings denoting the features
    Returns:
      enc_str - (str) binary string where 1 represents that corresponding 
                feature is present.
    eg: all_features = ["f1", "f2", "f3", "f4"]
        feature_list = ["f3", "f1"]
        Then, enc_str = 1010
    """
    enc_str = ""
    for feature in self.all_features:
      if feature in feature_list:
        enc_str += "1"
      else:
        enc_str += "0"
    return enc_str

  def decodeFeatures(self, enc_str):
    """
    Returns a list of features corresponding to the encoded_string.
    Args:
      enc_str  - (str) binary string
    Returns:
      feature_list - list of feature names corresponding to the encoded string.  
    """
    feature_list = []
    for i, feature in enumerate(self.all_features):
      if enc_str[i] == "1":
        feature_list.append(feature)
    return feature_list

  def getFitnessValue(self, enc_str):
    """
    Returns fitness value which signifies the performance of the encoded string
    """
    feature_list = decodeFeatures(enc_str)
    return self.obj_fn(feature_list)


####### Testing ########33
class A(SearchAlgo):
  def __init__(self, feature_list, obj_fn):
    super(A, self).__init__(feature_list, obj_fn)

  def print__(self):
    print(self.feature_list)

t = A([1,2,3], 5)
print(t.encodeFeatures([1, 2, 3]))
print(t.encodeFeatures([]))
print(t.decodeFeatures("010"))


