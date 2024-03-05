# Bank-Marketing-Predictive-learning
The data is related to direct marketing campaign (phone calls) from a banking institution in Portugal. The classification goal is to predict whether the client will subscribe to a term deposit. The dataset can be found in the Bank Marketing Dataset. Alhamdulillah, this project serves as the Final Project guided by dibimbing.id, a data science bootcamp institution where I study.<br>

![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/a97e808a-bc4a-4477-8645-c4faede26d07) ![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/470e667a-f14c-458a-a60e-79606aa60fc9)<br>

In case you're not aware, the .ipynb file contains both the code and explanations, which you can easily view [here](https://colab.research.google.com/drive/1gtapjhRExtyk6_o1DPGDHajJFLYesTIv#scrollTo=Kw-KOGOPW51I). Additionally, as this was my final project in a Data Science Bootcamp, I've provided a Google Slide Presentation summarizing all the procedures and findings. You can access it [here](https://docs.google.com/presentation/d/1YEwvO8cq5_Bf9o2mhKgGUGlbcU6Pck50/edit?usp=sharing&ouid=116801621305292269028&rtpof=true&sd=true).

## Objectives 
To be sure, banks can make money in a number of different ways, even if they are still essentially considered lenders. Generally, they make money by lending money to savers, who are then compensated with a certain interest rate and a guarantee of their funds. The borrowed money is then lent to the borrower who needs it at that time. However, the interest rate charged to borrowers is higher than the interest rate paid to depositors. The difference between the interest rate paid and the interest rate received is often called the spread, from which banks make a profit.<br>
The investigation focuses on a Portuguese banking institution that attempted to collect funds from depositors through a direct marketing campaign. Generally speaking, direct marketing campaigns require in-house or outsourced call centers. Although no information on cost of sales is provided, several articles note that this can significantly affect the cost-to-cost ratio of the product. In this case, the bank's sales team randomly contacted approximately 11,162 customers, so 52.6% of them (approximately 6540) were willing to make a deposit. <br>
However, the bank was looking for ways to help it run more effective marketing campaigns and improve conversion rates, and machine learning was one of the answers.

## Libraries
Libraries such as [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/) are the most commonly used in the analysis. However, I also used [sklearn](https://scikit-learn.org/stable/) to conduct the predictive analysis with some classification models.
```python
======Pandas Config========
import pandas as pd
pd.set_option("display.max_columns",None)

=======Numpy=========
import numpy as np

=======Visualization======
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
sns.set_palette("bright")
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

=======Preprocessing=======
# for Q-Q plots
import scipy.stats as stats
from feature_engine.outliers import Winsorizer
from scipy.stats import chi2_contingency

=========Modeling ===========
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

=====Warnings========
import warnings
warnings.filterwarnings("ignore")
```

## Success Criteria Model 
The percentage of customers that can be predicted for acquisition is at least 50%.

## Definisi Model Dan Baseline Model
- Definisi Model <br>
Classifikasi customer potensial untuk di akuisisi berdasarkan demographics, spend dan engagement.<br>
- Baseline Model <br>
![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/3afffda6-52d4-4478-bf89-148d52ec5b9e) <br>
There were 5871 customers who were acquired and 5291 customers who refused to be acquired.

## Data Cleaning
- Check Missing Value <br>
<img width="248" alt="image" src="https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/89b99037-4864-45d9-b631-196cba2d9acb">

The data type for each column is appropriate and there is no missing data

## Check Ouliers
<img width="583" alt="image" src="https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/beb4262c-ab68-487f-9623-c2f3988b7ae7"> <br>
Some columns have outliers but this is normal because the outliers are values ​​that are considered normal, such as age and also the number of times the bank contacted the customer.

## Eksplonatory Data Analysis

### Data Distribution
What is the distribution of the customers' ages? Which age group tends to have the highest likelihood of making deposits? <br>
![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/a3b9fafc-ea39-4928-b64d-d750632c79f0)<br>
The age distribution of the customers follows a normal distribution. <br>

![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/4ae41664-c27a-4541-8a74-2531e3421a93)<br>
By median, it appears that customers who make deposits are older compared to customers who do not deposit.

### Analisis
![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/5bc149ba-d40e-43eb-9bae-db4227d38b57)<br>
Customers who make deposits tend to have a moderate balance or balances below 40000, with a frequency below 20.

## Data Preprocessing
![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/671e4293-9b68-4c75-92d3-a4938bb47bc7)<br>
- Handling Multicolinearity <br>
![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/6d7f2891-6838-4f3c-8bbc-12637f5d66ed)<br>
Using the heat map correlation method, features such as 'pdays', 'previous', and 'poutcome' exhibit high correlation. However, removing these features enhances the model's performance by reducing multicollinearity or overfitting issues. This reduces the complexity of the model and improves its predictive ability.<br>
<img width="182" alt="image" src="https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/bf2987ca-cbb0-4001-b3f3-45903e402db2">
<img width="185" alt="image" src="https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/cf113a06-3f2d-4c8d-8875-f6bf37ae0aaf">
<img width="413" alt="image" src="https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/7b1d288a-b481-43a2-b4ba-0e1550219945">
<img width="412" alt="image" <img width="358" alt="image" src="https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/7351c726-8890-4db3-adbc-254ec4948b40">

## Modelling
![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/92ecc049-aa68-4ee4-8093-7b0a1d1439cf)
Random Forest will be selected for tuning. Random Forest will be chosen for optimization.

## Factor Importance
![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/78abe757-63cd-479f-9ca9-686e0dff4152)
From the plot, 3 features with the most contribution are sos-con variables balance, day and month. 

## CUMULATIVE GAINS CURVE
![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/7fc39342-3af0-4546-8314-91a7a6b91ffa)
Based on cumulative gains analysis, the model demonstrates good performance by successfully identifying 70% of customers who are likely to open a deposit account when focused on the top 20% of the population with the highest probability. Therefore, out of 10,000 customers, 2,000 customers are prioritized.

## LIFT CURVE
![image](https://github.com/Nurgi2512/Bank-Marketing-Predictive-learning/assets/147684817/90640e9c-e197-4104-944d-cebac897e67b)
The model performs 1.4 times better than random choice on the top 20% of the population.















