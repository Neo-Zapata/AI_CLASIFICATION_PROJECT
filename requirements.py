import random
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math
from rich.console import Console
from rich.table import Table
from rich.style import Style
from rich.text import Text
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tabulate import tabulate
import warnings
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore", message="elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison")

cwd = os.getcwd()
console = Console()

N_FEATURES = 20
EPSILON = 1e-8
N_CLASSES = 3
NORMAL = 1
SUSPECT = 2
PATHOLOGIC = 3