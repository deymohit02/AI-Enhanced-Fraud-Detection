import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import sklearn.metrics as mt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

try:
    import seaborn as sns
    import plotly.express as px
    from matplotlib import pyplot as plt
except ImportError:
    pass

all_performances = pd.DataFrame()
list_clf_name = []
list_pred = []
list_model = []

def fit_model(model, X_train, y_train):
    X_model = model.fit(X_train,y_train)
    return X_model

def add_list(name, model, y_pred):
    global list_clf_name, list_pred, list_model, list_x_test
    list_clf_name.append(name)
    list_model.append(model)
    list_pred.append(y_pred)


def add_all_performances(name, precision, recall, f1_score, AUC):
    global all_performances
    new_row = pd.DataFrame([[name, precision, recall, f1_score, AUC]],
                         columns=["model_name","precision", "recall", "f1_score", "AUC"])
    all_performances = pd.concat([all_performances, new_row], ignore_index=True)
    all_performances = all_performances.drop_duplicates()
      
    
def calculate_scores(X_train, X_test, y_train, y_test, y_pred, name, model):
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = mt.precision_score(y_test,y_pred)
    recall = mt.recall_score(y_test,y_pred)
    f1_score= mt.f1_score(y_test, y_pred)
    
    # CRITICAL FIX: Calculate AUC using probabilities, not binary predictions
    # This was causing AUC ≈ 0.5 (random guessing)
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        AUC = roc_auc_score(y_test, y_pred_proba)
    except AttributeError:
        # Fallback for models without predict_proba
        AUC = roc_auc_score(y_test, y_pred)
    
    add_list(name, model, y_pred)
    add_all_performances(name, precision, recall, f1_score, AUC)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1_score, "auc": AUC}
    

def model_performance(model, X_train, X_test, y_train, y_test, technique_name):

    name= model.__class__.__name__+"_"+technique_name
    x_model = fit_model(model, X_train, y_train)
    y_pred = x_model.predict(X_test)
    print("***** "+ name +" DONE *****")

    metrics = calculate_scores(X_train, X_test, y_train, y_test, y_pred, name, model)
    return x_model, metrics
    
    
def display_all_confusion_matrices(y_test):
    column = 4
    row = int(all_performances["model_name"].count()/4)
    f, ax = plt.subplots(row, column, figsize=(20,10), sharey='row')
    ax = ax.flatten()

    for i in range(column*row):
        cf_matrix = confusion_matrix(y_test, list_pred[i])
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot(ax=ax[i], xticks_rotation=45)
        disp.ax_.set_title(list_clf_name[i]+"\nAccuracy:{accuracy:.4f}\nAUC:{auc:.4f}"
                           .format(accuracy= accuracy_score(y_test, list_pred[i]),auc= roc_auc_score(y_test, list_pred[i])),
                             fontsize=14)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i!=0:
            disp.ax_.set_ylabel('')


    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust()
    f.colorbar(disp.im_)
    plt.show()
