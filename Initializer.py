from requirements import *



def Initializer(N = N_FEATURES):
    training_df = pd.read_csv(cwd + "/dataset/training.csv")
    testing_df = pd.read_csv(cwd + "/dataset/testing.csv")
    # training_df = training_df.drop(training_df.columns[0], axis=1) # to remove row indexes
    # testing_df = testing_df.drop(testing_df.columns[0], axis=1) # to remove row indexes
    
    x_train = training_df.iloc[:, :N].values # (n,m)
    y_train = training_df.iloc[:, -1].values # (n,)
    x_test = testing_df.iloc[:, :N].values # (n,m)

    if N == 1:
        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)    # (n,1)

    # x_train = np.where(x_train == 0, EPSILON, x_train)
    # x_test = np.where(x_test == 0, EPSILON, x_test)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    X_train = np.insert(x_train, 0, 1, axis = 1) # adding the bias to X as first column
    X_test = np.insert(x_test, 0, 1, axis = 1) # adding the bias to X as first column

    np.set_printoptions(precision = 5, suppress = True)

    return X_train, X_test, y_train