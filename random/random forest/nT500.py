# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pdb
from helpers import perfMeasures
import sys    
import os    
file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
print(file_name)

# File Paths
data_path = "normfeat_data.csv"
train_path = "C:/Users/Chris Lo/Downloads/train.csv"
test_path = "C:/Users/Chris Lo/Downloads/test.csv"
output_path = "C:/Users/Chris Lo/Downloads/labels.csv"

def get_headers(dataset):
    return dataset.columns.values

def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset

def split_dataset(dataset, train_percentage, feature_headers, label_header):
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[label_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def handle_missing_values(dataset, missing_values_header, missing_label):
    return dataset[dataset[missing_values_header] != missing_label]

def random_forest_classifier(features, label):
    clf = RandomForestClassifier(n_estimators=500, max_depth=50, min_samples_leaf=2, max_features=0.1, n_jobs=-1,
                                random_state=0)
    clf.fit(features, label)
    return clf

def dataset_statistics(dataset):
    print(dataset.describe())

def dropFeatures(headers, dataset, threshold):
    rf = RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_leaf=4, max_features=0.1, n_jobs=-1,
                                random_state=0)
    rf.fit(dataset.drop(['id', 'label'], axis=1), dataset.label)
    unimportantFeatures = []
    for i in range(len(rf.feature_importances_)):
        if rf.feature_importances_[i] < threshold:
            unimportantFeatures.append(i)
    dataset.drop(dataset.columns[unimportantFeatures], inplace=True, axis=1)
    np.delete(headers, unimportantFeatures)
    return dataset
    new_headers = headers
    return headers
    return new_headers

def main():
    # Load the csv file into pandas dataframe
    data = pd.read_csv(data_path)
    where_are_NaNs = np.isnan(data)
    data[where_are_NaNs] = 0

    num_features = len(data.iloc[1, :]) - 2

    # Get headers and add attributes to headers of dataset
    headers = get_headers(data)
    print(headers)
    add_headers(data, headers)
    # Get basic statistics of the loaded dataset
    dataset_statistics(data)
    # Drop unwanted features
    #dropFeatures(headers, data, 0.01)

    # Split data into training and testing data
    train_x, test_x, train_y, test_y = split_dataset(data, 0.67, headers[2:], headers[1])
    # Alternatively, simply load train_data and test_data
    # insert code here

    pca = PCA(num_features).fit(train_x)
    components = pca.transform(train_x)
    train_x = pca.inverse_transform(components)

    pca = PCA(num_features).fit(test_x)
    components = pca.transform(test_x)
    test_x = pca.inverse_transform(components)

    # Train and test dataset shapes
    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)

    # Create random forest classifier instance
    trained_model = random_forest_classifier(train_x, train_y)
    print("Trained model :: ", trained_model)
    predictions = trained_model.predict(test_x)
    pred = trained_model.predict_proba(test_x)[:,1]

    p1 = pred
    n = 1000
    lbl = np.array(test_y)
    dic = perfMeasures(p1, n, lbl, nm = file_name)


    import pickle
    def save_obj(obj, file_name ):
        with open('out/'+ file_name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)
    

    # Model Accuracy
    for i in range(0, 5):
        print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
    trnA = accuracy_score(train_y, trained_model.predict(train_x))
    tstA = accuracy_score(test_y, predictions)
    conf = confusion_matrix(test_y, predictions)
    print("Train Accuracy :: ", trnA)
    print("Test Accuracy  :: ", tstA)
    print(" Confusion matrix ", conf)
    dic['trnA']=trnA
    dic['tstA']=tstA
    dic['conf']=conf
    save_obj(dic, file_name )

    """prediction_data = pd.read_csv(test_path)
    prediction_id = pd.DataFrame(prediction_data.ix[:,0])
    prediction_y = prediction_data.ix[:,1:]
    final_prediction = pd.DataFrame(trained_model.predict(prediction_y))

    prediction = prediction_id.join(final_prediction)
    prediction.to_csv(output_path)"""

    # VISUALIZATION OF A TREE IN THE RANDOM FOREST

    # Import tools needed for visualization

    """
    from sklearn.tree import export_graphviz
    import pydot
    # Pull out one tree from the forest
    tree = trained_model.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = headers, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree.png')
    """

if __name__ == "__main__":
    main()