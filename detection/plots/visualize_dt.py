import joblib
from sklearn import tree
import matplotlib.pyplot as plt
from six import StringIO
from PIL import Image
import pydotplus
import pandas as pd

"""
This file is used to visualize the decision tree of the random forest classifier. It works also for DTs and GBs.
"""


clf = joblib.load("data/models/rf.pkl")
dot_data = StringIO()
"""tree.export_graphviz(clf.named_steps['classifier'], out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)"""

chosen_tree = clf.named_steps["classifier"].estimators_[0]
validation = pd.read_csv("data/generated data/preprocessing/validation.csv") # load validation set
validation = validation.drop(columns=["POTENTIAL_FRAUD"]) # drop columns that are not needed for classification # "RATIO_SECTIO"
feature_names = list(validation.columns) # get feature names from validation set
del(validation)
tree.export_graphviz(
    chosen_tree, #clf.named_steps["classifier"], #
    out_file=dot_data,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    special_characters=True,
)

# Convert to PNG using pydotplus
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_svg("random_forest.svg")
