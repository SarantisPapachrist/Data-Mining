from pgmpy.models import BayesianNetwork
from pgmpy.estimators import K2Score,HillClimbSearch,MaximumLikelihoodEstimator
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D
from keras.optimizers import Adam

##Bayesian NetWork Creation
def bay_net_structure(data):
    x = data[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']]
    y = data['labels']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test,y_test], axis=1)

    est = HillClimbSearch(train_data)
    model_structure = est.estimate(scoring_method=K2Score(train_data), max_indegree=2, epsilon=0.001)
    
    print("Learned structure edges:", model_structure.edges())
    
    # Fit the Bayesian Network model
    model = BayesianNetwork(model_structure)
    model.fit(train_data,estimator=MaximumLikelihoodEstimator)
    
    print("Model CPDs:", model.get_cpds())
    
    test_data_combined = test_data.copy()
    test_data_combined['labels'] = y.loc[test_data.index]
    predictions = model.predict(test_data_combined.drop(columns='labels'))
    accuracy = accuracy_score(test_data_combined['labels'], predictions['labels'])
    f1=f1_score(test_data_combined['labels'], predictions['labels'],average='weighted')
    print("Predictive Accuracy:", accuracy)
    print("F1 score:", f1)
    
    return model, accuracy


#Random Forest Algorithm
def random_forest(data,labels):
    x=data
    y =labels
    y=labels
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    rf = RandomForestClassifier(criterion='entropy',min_samples_leaf=50,n_estimators=200)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred,average='weighted')
    print("Accuracy:", accuracy)
    print("F1 score:", f1)
    for i in range(3):
        tree = rf.estimators_[i]
        dot_data = export_graphviz(tree,
                                feature_names=X_train.columns,  
                                filled=True,  
                                max_depth=2, 
                                impurity=False, 
                                proportion=True)
        graph = graphviz.Source(dot_data)
        return graph



#2D Convolutional Neural Network
def nn(data,labels):
    x=data
    y=labels-1
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print(X_train.shape)
    print(y_train.shape)
   
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation = 'relu', input_shape = X_train[0].shape))
    

    model.add(Dropout(0.1))

    model.add(Conv2D(64, (2, 2), activation='relu'))
    

    model.add(Dropout(0.2))

    model.add(Conv2D(128, (2, 2), activation='relu'))
    
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(12, activation='softmax'))
        

    
    model.compile(optimizer=Adam(learning_rate = 0.001), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    history = model.fit(X_train, y_train,batch_size=32, epochs = 10, validation_data= (X_test, y_test), verbose=1)
    return model, history