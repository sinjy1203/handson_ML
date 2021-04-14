from sklearn.tree import export_graphviz
import os
from graphviz import Source

print(Source.from_file(os.path.join("C:/Users/sinjy/jupyter_notebook/datasets", "iris_tree.dot")))