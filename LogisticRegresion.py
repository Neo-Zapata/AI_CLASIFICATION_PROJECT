from requirements import *


def gen_w(N):
    w = [np.random.rand() for i in range(0, N)]
    w = np.array(w)
    w = w.reshape((N, 1))
    return w  

def gen_bucket(X):
    n = len(X)
    # B1 = int(2**(math.log10(n**2)))
    B1 = int(n*0.1)
    return B1

def join(normal, suspect, pathologic):
    matrix = np.column_stack([normal, suspect, pathologic])
    max_indices = np.argmax(matrix, axis=1)
    # due indexes starting from 0 are just 1 value phased from the actual class value, we add 1
    max_indices += 1
    return max_indices


class LogisticRegresion:
    def __init__(self, epochs, alpha) -> None:
        self.W = 0
        self.epochs = epochs
        self.alpha = alpha

    def fit(self, X, y, i, display=False):
        self.Train(X, y, display, i)

    def predict(self, X):
        prediction = self.Sigmoid(X)
        rounded = np.round(prediction).astype(int)
        prediction = np.where(rounded > 0.5, 1, 0)
        return prediction
    
    def score(self, X, y):
        prediction = self.predict(X)
        accuracy = np.sum(prediction == y) / len(y)
        return accuracy
    
    # def Get_Error(self, X, y):
    #     L = self.Loss(X, y)
    #     print(L)
    #     return round(L, 5)

    def Train(self, X, y, display, classe):
        self.w = gen_w(X.shape[1])
        bucket = gen_bucket(X)
        Losses = []
        for _ in range(0, self.epochs):
            sample_rows = np.random.choice(X.shape[0], size = bucket, replace = False)
            cX = X[sample_rows, :]
            cy = y[sample_rows, :]
            Losses.append(self.Loss(cX, cy))
            self.Update(cX, cy)
        if display:
            self.Display(Losses, classe)
        if X.shape[1] == 1 or X.shape[1] == 2:
            self.Pretty_plot(X, y)

    def Display(self, L, classe):
        if classe == 1:
            classe = "Normal"
        elif classe == 2:
            classe = "Suspect"
        else:
            classe = "Pathologic"
        
        console.print(f"\n[magenta]------------------ Displaying for {classe} model ------------------[magenta]", justify="center")
        bias = Text("Bias: ")
        bias.stylize("bold magenta")
        console.print(bias, justify="center")
        console.print(f"[cyan]{self.w[0]}[cyan]\n", justify="center")

        w = Text("W: ")
        w.stylize("bold magenta")
        console.print(w, justify="center")    
        console.print(f"[cyan]{tabulate([self.w], headers=['Vector'], tablefmt='grid')}[cyan]", justify="center")

        # plt.clf()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(L, color="red", label="Error")
        ax.set_yscale('log')
        ax.set_xlabel('Time')
        ax.set_ylabel('Error')
        ax.set_title('Error vs. Time')
        ax.legend()
        fig.savefig(cwd + f'/LogReg_results/LR_Error_{classe}.png', dpi=300)
        # plt.show()
        plt.close()
        # print error

    def Hiperplano(self, X):
        Hiperplano =  np.dot(X, self.w)
        return Hiperplano

    def Sigmoid(self, X):
        Hiperplano  = self.Hiperplano(X)
        Sigmoid     = 1 / (1 + math.e**(-Hiperplano))
        return Sigmoid

    def Loss(self, X, y):
        n           = len(X)
        Sigmoid     = self.Sigmoid(X)
        Loss = - (1 / n) * np.sum(y * np.log(Sigmoid + EPSILON) + (1 - y) * np.log((1 - Sigmoid) + EPSILON))
        return Loss

    def Derivatives(self, X, y):
        y_pred      = self.Sigmoid(X)
        y_rest      = y - y_pred
        n           = len(y) 
        dw = (1/(n)) * np.matmul(np.transpose(y_rest), -X)
        return dw

    def Update(self, X, y):
        dw = self.Derivatives(X, y)
        self.w = self.w - (self.alpha * (dw.T))
        


# The accuracy of a binary classification model can be evaluated using a confusion matrix. A confusion matrix is a table that summarizes the number of correct and incorrect predictions made by the model.

# The four possible outcomes are:

# True positives (TP): correctly predicted positive instances.

# False positives (FP): predicted positive instances when they are actually negative.

# True negatives (TN): correctly predicted negative instances.

# False negatives (FN): predicted negative instances when they are actually positive.

# From the confusion matrix, several metrics can be computed:

# Accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision = TP / (TP + FP)

# Recall = TP / (TP + FN)

# F1 score = 2 * (precision * recall) / (precision + recall)

# The accuracy is the ratio of correctly predicted instances to the total number of instances. The precision measures the model's ability to correctly identify positive instances, while the recall measures the model's ability to identify all positive instances. The F1 score is the harmonic mean of precision and recall, and it provides a single metric that balances both measures.

