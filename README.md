# Bank-Marketing-Predictive-learning
The data is related to direct marketing campaign (phone calls) from a banking institution in Portugal. The classification goal is to predict whether the client will subscribe to a term deposit. The dataset can be found in the Bank Marketing Dataset. Alhamdulillah, this project serves as the Final Project guided by dibimbing.id, a data science bootcamp institution where I study.

![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/a97e808a-bc4a-4477-8645-c4faede26d07) ![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/470e667a-f14c-458a-a60e-79606aa60fc9)

In case you're not aware, the .ipynb file contains both the code and explanations, which you can easily view [here](https://colab.research.google.com/drive/1gtapjhRExtyk6_o1DPGDHajJFLYesTIv#scrollTo=Kw-KOGOPW51I). Additionally, as this was my final project in a Data Science Bootcamp, I've provided a Google Slide Presentation summarizing all the procedures and findings. You can access it [here](https://docs.google.com/presentation/d/1YEwvO8cq5_Bf9o2mhKgGUGlbcU6Pck50/edit?usp=sharing&ouid=116801621305292269028&rtpof=true&sd=true).

## Objectives 
To be sure, banks can make money in a number of different ways, even if they are still essentially considered lenders. Generally, they make money by lending money to savers, who are then compensated with a certain interest rate and a guarantee of their funds. The borrowed money is then lent to the borrower who needs it at that time. However, the interest rate charged to borrowers is higher than the interest rate paid to depositors. The difference between the interest rate paid and the interest rate received is often called the spread, from which banks make a profit.

The investigation focuses on a Portuguese banking institution that attempted to collect funds from depositors through a direct marketing campaign. Generally speaking, direct marketing campaigns require in-house or outsourced call centers. Although no information on cost of sales is provided, several articles note that this can significantly affect the cost-to-cost ratio of the product. In this case, the bank's sales team randomly contacted approximately 11,000 customers, so 52.6% of them (approximately 5,800) were willing to make a deposit

However, the bank was looking for ways to help it run more effective marketing campaigns and improve conversion rates, and machine learning was one of the answers.

## Libraries
Libraries such as [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/) are the most commonly used in the analysis. However, I also used [sklearn](https://scikit-learn.org/stable/) to conduct the predictive analysis with some classification models.
```python
#======Pandas Config========
import pandas as pd
pd.set_option("display.max_columns",None)

#=======Numpy=========
import numpy as np

#=======Visualization======
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
sns.set_palette("bright")
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

#=======Preprocessing=======
# for Q-Q plots
import scipy.stats as stats
from feature_engine.outliers import Winsorizer
from scipy.stats import chi2_contingency

#=========Modeling ===========
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn import svm,tree
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc,roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split as tts
from sklearn import model_selection
#=====Warnings========
import warnings
warnings.filterwarnings("ignore")


#======Function========
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def create_stacked_bar_percent(df,column_name):

    # Get the percentage of default by each group
    default_by_group = pd.crosstab(index=df['deposit'],columns = df[column_name], normalize = 'columns')
    default_by_group = default_by_group[default_by_group.iloc[1].sort_values().index]

    # Round up to 2 decimal
    default_by_group = default_by_group.apply(lambda x: round(x,2))

    labels = default_by_group.columns
    list1 = default_by_group.iloc[0].to_list()
    list2 = default_by_group.iloc[1].to_list()

    list1_name = "deposit"
    list2_name = "Not Deposit"
    title = f" %Deposit by {column_name}"
    xlabel = column_name
    ylabel = "Number of Deposit"

    fig, ax = plt.subplots(figsize=(8,8),dpi=100)
    bar_width = 0.5

    ax1 = ax.bar(labels,list1, bar_width, label = list1_name)
    ax2 = ax.bar(labels,list2, bar_width, bottom = list1, label = list2_name)

    ax.set_title(title, fontweight = "bold",fontsize=12)
    ax.set_xlabel(xlabel, fontweight = "bold",fontsize=12)
    ax.set_ylabel(ylabel, fontweight = "bold",fontsize=12)
    ax.legend(loc="upper right")

    plt.xticks(list(range(len(labels))), labels,rotation=90)
    plt.yticks(fontsize=12)

    for r1, r2 in zip(ax1, ax2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., f"{h1:.0%}", ha="center", va="center", color="black", fontsize=12, fontweight="bold")
        plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., f"{h2:.0%}", ha="center", va="center", color="black", fontsize=12, fontweight="bold")

    plt.show()

def diagnostic_plots(df, variable):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('RM quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()

def missing_check(df):
  missing = df.isnull().sum()
  missing_per = round(missing/len(df),4)*100
  unique_val = df.nunique()
  type_data = df.dtypes
  df = pd.DataFrame({'Missing_values':missing,
                    'Percent of Missing (%)':missing_per,
                    'Numbers of Unique':unique_val,
                    'Data type':type_data})
  return df
