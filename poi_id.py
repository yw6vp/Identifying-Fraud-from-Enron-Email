#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier as DTC

# print the a list
def clean_print(array):
    for line in array:
        print line
        
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))
print "Total number of entries:", len(data_dict.keys())
print "Total number of features:", len(data_dict[data_dict.keys()[0]])
# Count how many missing values are there for each feature
all_features = data_dict[data_dict.keys()[0]].keys()
missing_values = {}
# Initialize the dictionary
for feature in all_features:
    missing_values[feature] = 0
for key in data_dict.keys():
    for feature in all_features:
        if data_dict[key][feature] == 'NaN':
            missing_values[feature] += 1
missing_values_list = missing_values.items()
missing_values_list.sort(key = lambda x: x[1])
# Print the number of missing values for each feature
print "Number of missing values for all the features:"
clean_print(missing_values_list)
print 
### Remove "TOTAL" and "THE TRAVEL AGENCY IN THE PARK" from dictionary
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# first selcet all features then decide which features are more important
features_list = all_features 
features_list.remove('poi')
# Make a deep clone and insert 'poi'
features_list_with_poi = [feature for feature in features_list]
features_list_with_poi.insert(0, 'poi')
# get rid of features that have string values
remove_features = ['email_address']
for remove_feature in remove_features:
    features_list.remove(remove_feature)
    features_list_with_poi.remove(remove_feature)
    
# Get total number of features
n_of_features = len(features_list)

# Call featureFormat to split features and labels
data_array = featureFormat(data_dict, features_list_with_poi)
labels, features = targetFeatureSplit(data_array)

# Set k to use SelectKBest
# I've adjusted k to get the best precision and recall when running test_classifier
kbest = 5
feature_selector = SelectKBest(f_classif, k = kbest)
feature_selector.fit(features, labels)
# Sort features according to scores
scores = feature_selector.scores_
features_scores = zip(features_list, scores)
features_scores.sort(key = lambda x: x[1], reverse = True)
# Print out features and their scores
print "Features sorted by their importances:"
for pair in features_scores:
    print pair
# Transform the features, use only the k most important features
features_transformed = feature_selector.transform(features)
# Get the names of the selected features
top_features = [pair[0] for pair in features_scores[0:kbest]]
new_features_list = [feature for feature in features_list if feature in top_features]
features_list = new_features_list
# Add the email features for testing
features_list.append('from_this_person_to_poi')
features_list.append('from_poi_to_this_person')
# The first feature needs to be 'poi'
features_list_with_poi = [feature for feature in features_list]
features_list_with_poi.insert(0, 'poi')

### Add new features: 'from_this_person_to_poi_percentage', 'from_poi_to_this_person_percentage'
for person in data_dict.values():
    if person["from_messages"] != "NaN" and person["from_this_person_to_poi"] != "NaN":
        person['from_this_person_to_poi_percentage'] = float(person['from_this_person_to_poi']) / person['from_messages']
    else:
        person['from_this_person_to_poi_percentage'] = 0
        
    if person['to_messages'] != 'NaN' and person['from_poi_to_this_person'] != 'NaN':
        person['from_poi_to_this_person_percentage'] = float(person['from_poi_to_this_person']) / person['to_messages']
    else:
        person['from_poi_to_this_person_percentage'] = 0
    
### Without using ratio, Precision: 0.41920    Recall: 0.34500
### With ratio, Precision: 0.44003    Recall: 0.34300
### Without any email features, precision = 0.49830, recall = 0.36650
### Not using email features gives the best result
features_list.remove('from_this_person_to_poi')
features_list.remove('from_poi_to_this_person')
features_list_with_poi.remove('from_this_person_to_poi')
features_list_with_poi.remove('from_poi_to_this_person')
#features_list.extend(['from_this_person_to_poi_percentage', 'from_poi_to_this_person_percentage'])
#features_list_with_poi.extend(['from_this_person_to_poi_percentage', 'from_poi_to_this_person_percentage'])

# remove outliers
# Define a function for sorting
def sort_features_array(i = 0):
    if i >= len(features_array[0]):
        i = 0
    array = zip(*features_array)
    array = zip(array[i], array[-1])
    return sorted(array, key = lambda x: x[0])
        
### Task 2: Remove outliers
keys = data_dict.keys()
features_array = []
for key in keys:
    person = []
    for feature in features_list:
        person.append(data_dict[key][feature])
    person.append(key)
    features_array.append(person)
#print "*************sorted according to feature 0: salary****************"
#sorted_array = sort_features_array(0)
#clean_print(sorted_array)
#print "*************sorted according to feature 1: exercised_stock_options****************"
#sorted_array = sort_features_array(1)
#clean_print(sorted_array)
#print "*************sorted according to feature 2: bonus****************"
#sorted_array = sort_features_array(2)
#clean_print(sorted_array)
#print "*************sorted according to feature 3: total_stock_value****************"
#sorted_array = sort_features_array(3)
#clean_print(sorted_array)
#print "*************sorted according to feature 4: deferred_income****************"
#sorted_array = sort_features_array(4)
#clean_print(sorted_array)
# remove people with strange values 
# 'BANNANTINE JAMES M' and 'GRAY RODNEY' have salaries that are too low
# 477 and 6615 versus at least 60000 for everyone else
# 'BELFER ROBERT' and 'GILLIS JOHN' both have very low exercised_stock_options
# 3285 and 9803 versus above 17000+ for the rest
# 'BELFER ROBERT' is also the only one with negative total_stock_value
### The following list comes from the above analysis
remove_names = ['BANNANTINE JAMES M', 'GRAY RODNEY', 'BELFER ROBERT', 'GILLIS JOHN']
for name in remove_names:
    if name in data_dict:
        data_dict.pop(name)

my_dataset = data_dict

### Extract features and labels from dataset for local testing
# The function featureFormat also removes one point whose features are all zeroes
print "features: ", features_list_with_poi
data_array = featureFormat(data_dict, features_list_with_poi)
labels, features = targetFeatureSplit(data_array)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Create Naive Bayes classifier and fit
# For the dataset, Accuracy: 0.86667, Precision = 0.5, Recall = 0.38889
# For ssscv, Accuracy: 0.85679, Precision: 0.49830, Recall: 0.36650
clf = GaussianNB()

# Create decision tree classifier and fit
# Before adjusting parameters, 
# For the dataset, Accuracy : 1.0, Precision: 1.0, Recall: 1.0
# For ssscv Accuracy: 0.79064, Precision: 0.25120, Recall: 0.23500
#clf = DTC(random_state = 42)
# After adjusting parameters, 
# For the dataset, Accuracy: 0.97778, Precision: 1.0, Recall: 0.83333
# For ssscv, Accuracy: 0.83793, Precision: 0.42804, Recall: 0.40000
# max_features turns out to be the most important change
#clf = DTC(criterion = 'gini', max_depth = 7, min_samples_split = 2, max_features = 3, random_state = 42)

clf.fit(features, labels)
pred = clf.predict(features)
print "Accuracy for the dataset:", clf.score(features, labels)
print "Precision for the dataset:", precision_score(labels, pred)
print "Recall for the dataset:", recall_score(labels, pred)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list_with_poi)
### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)