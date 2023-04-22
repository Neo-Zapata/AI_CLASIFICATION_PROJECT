from LogisticRegresion import LogisticRegresion, join
from Initializer import Initializer
from requirements import *

random.seed(42) # "Answer to the Ultimate Question of Life, the Universe, and Everything"

# Initializer accepts 1 optional parameter which is the # of features to use (counting from the begining). 
# By default, it uses all features.
X, X_test, y = Initializer()

def logistic_regresion(epochs, alpha):
    kFold_accuracy = []
    AUC = []
    kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
    last_y_true = last_y_pred = 0
    for train_index, test_index in kf.split(X):        
        # split into training and testing
        cX_train, cX_test = X[train_index], X[test_index]
        cy_train, cy_test = y[train_index], y[test_index]
        last_y_true = cy_test

        # Mini-Batch Gradient Descent
        # One-vs-All method
        LR_model_normal     = LogisticRegresion(epochs, alpha)
        LR_model_suspect    = LogisticRegresion(epochs, alpha)
        LR_model_pathologic = LogisticRegresion(epochs, alpha)

        y_binary_normal     = (cy_train == 1).astype(int)
        y_binary_suspect    = (cy_train == 2).astype(int)
        y_binary_pathologic = (cy_train == 3).astype(int)

        ty_binary_normal     = (cy_test == 1).astype(int)
        ty_binary_suspect    = (cy_test == 2).astype(int)
        ty_binary_pathologic = (cy_test == 3).astype(int)

        LR_model_normal.fit(cX_train, y_binary_normal, 1, display=False)
        LR_model_suspect.fit(cX_train, y_binary_suspect, 2, display=False)
        LR_model_pathologic.fit(cX_train, y_binary_pathologic, 3, display=False)
        
        # Predictions (percentage)
        normal_score      = LR_model_normal.score(cX_test, ty_binary_normal) 
        suspect_score     = LR_model_suspect.score(cX_test, ty_binary_suspect) 
        pathologic_score  = LR_model_pathologic.score(cX_test, ty_binary_pathologic) 

        accuracy = round((normal_score + suspect_score + pathologic_score) / N_CLASSES, 3)

        kFold_accuracy.append(accuracy)

        normal_sigmoid      = LR_model_normal.Sigmoid(cX_test)
        suspect_sigmoid     = LR_model_suspect.Sigmoid(cX_test)
        pathologic_sigmoid  = LR_model_pathologic.Sigmoid(cX_test)

        last_y_pred         = join(normal_sigmoid, suspect_sigmoid, pathologic_sigmoid)
        last_y_pred_matrix  = np.column_stack([normal_sigmoid, suspect_sigmoid, pathologic_sigmoid])
        scaled_matrix = last_y_pred_matrix / last_y_pred_matrix.sum(axis = 1, keepdims = True)

        AUC.append(round(roc_auc_score(cy_test, scaled_matrix, multi_class='ovr'), 5))


    target_names = ['1', '2', '3']
    console.print("\n[magenta]AUC (one-vs-rest): [magenta]")
    console.print(round(np.mean(AUC), 5))
    console.print("\n[magenta]Classification Report: [magenta]")
    console.print(f"[cyan]{classification_report(last_y_true, last_y_pred, target_names=target_names)}[cyan]")

    KFold_mean_accuracy_score = round(sum(kFold_accuracy) / len(kFold_accuracy), 4) # in percentage
    console.print("[magenta]KFold Mean Accuracy Score:[magenta]")
    console.print(f"[cyan]{KFold_mean_accuracy_score}[cyan]\n")

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    n_bootstraps = 50
    bootstraps_scores = []

    bX_train, bX_test, by_train, by_test = train_test_split(X, y, test_size=0.2, random_state=42)

    LR_model_normal     = LogisticRegresion(epochs, alpha)
    LR_model_suspect    = LogisticRegresion(epochs, alpha)
    LR_model_pathologic = LogisticRegresion(epochs, alpha)
    
    for i in range(n_bootstraps):
        # Generate a bootstrap sample with replacement
        X_boot, y_boot = resample(bX_train, by_train, random_state = i)

        y_binary_normal     = (y_boot == 1).astype(int)
        y_binary_suspect    = (y_boot == 2).astype(int)
        y_binary_pathologic = (y_boot == 3).astype(int)

        LR_model_normal.fit(X_boot, y_binary_normal, 1, display=False)
        LR_model_suspect.fit(X_boot, y_binary_suspect, 2, display=False)
        LR_model_pathologic.fit(X_boot, y_binary_pathologic, 3, display=False)

        ty_binary_normal     = (by_test == 1).astype(int)
        ty_binary_suspect    = (by_test == 2).astype(int)
        ty_binary_pathologic = (by_test == 3).astype(int)

        normal_approx      = LR_model_normal.predict(bX_test) 
        suspect_approx     = LR_model_suspect.predict(bX_test) 
        pathologic_approx  = LR_model_pathologic.predict(bX_test) 

        # calculate accuracy score
        normal_score = accuracy_score(ty_binary_normal, normal_approx)
        suspect_score = accuracy_score(ty_binary_suspect, suspect_approx)
        pathologic_score = accuracy_score(ty_binary_pathologic, pathologic_approx)

        final_score = (normal_score + suspect_score + pathologic_score) / N_CLASSES

        bootstraps_scores.append(final_score)
    
    # calculate and print bootstrap statistics
    console.print("[magenta]Bootstrap Statistics:[magenta]")
    console.print(f"[cyan]Mean: {np.mean(bootstraps_scores):.3f}[cyan]")
    console.print(f"[cyan]Standard Error: {np.std(bootstraps_scores):.3f}[cyan]")

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




logistic_regresion(epochs = 1000, alpha = 0.15)
