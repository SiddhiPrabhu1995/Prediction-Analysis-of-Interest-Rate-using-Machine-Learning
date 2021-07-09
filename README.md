# Prediction Analysis of Interest Rate Using Machine Learning

CLAAT Document Link: https://codelabs-preview.appspot.com/?file_id=1G2ItLaq5UhDOsgHzytL7jwxbJWeX0_atEB0mXcQel30#7

Steps to Regenerate the Project:

1. Clone the Project into any directory of your choice
2. Create a sub-folder called "Mice Data" inside the Data folder
3. Download the Data from https://www.kaggle.com/wendykan/lending-club-loan-data and place it inside the "Original Data" in the Data folder
4. Check Folders Under the Code Folder
5. Understanding the Data: Notebook to Understand the Data more prominently
6. Cleansing, Preprocessing and EDA: Notebooks to Cleanse and Preprocess Data. , Notebook to impute Missing Values using MICE , Notebook to Normalize the Data
7. Feature Selection: Notebook to Implement Feature Tools , Notebook to Select Features using LassoCV
8. Models: This Folder Contains Sub-Folders with Notebooks to implement Linear Regression, Random Forests, Neural Networks and also implement AutoML using AutoSKLearm, H20.ai and Tpot

## INTRODUCTION:

![image](https://user-images.githubusercontent.com/57429405/125011620-cef5c080-e036-11eb-9da9-5d2bbd8ab11c.png)

Lending Club offers loans of various grades they assign that correspond to specific interest rates 
for investors. It is a peer-to-peer lending company, the largest of its kind in the world. Lending 
club aims to operate a platform at low cost to offer interest rates to our borrower members lower 
than the rates they could obtain through credit cards or traditional banks. The higher the interest 
rate, the riskier the grade. The risk comes in the form of defaults - whenever a loan defaults, 
investors end up losing a portion of their investment. On the basis of the borrower’s credit score, 
credit history, desired loan amount and the borrower’s debt-to-income ratio, Lending Club 
determines whether the borrower is credit worthy and assigns to its approved loans a credit grade 
that determines payable interest rate and fees. 

## OBJECTIVE:

![image](https://user-images.githubusercontent.com/57429405/125011645-d9b05580-e036-11eb-9a03-a67c6c653437.png)


1. To work on loan predictions using machine learning models and provide a model that caters 
their needs.
2. To provide a full funded loan from lending club to clients
3. To predict the lowest possible interest rates
4. To avoid loans that are predicted to default
5. To get a desired loan duration
6. To provide an online lending platform where borrowers are able to obtain loans and investors 
can purchase notes backed by payments based on loans
7. To predict the expected returns for loans to a given borrower
8. To maximize our returns by predicting the probability of default of the borrower so as to help 
avoid investment in those high-risk notes

## 1. ANALYSING THE DATASET

Visualize the various features of dataset using python libaries.

<img width="325" alt="ANALYZEDATA" src="https://user-images.githubusercontent.com/57429405/125010884-6a863180-e035-11eb-9880-c0009d0e47f2.PNG">

## 2. CLEANSING AND PREPROCESSING

### 1. Importing the Data and removing unnecessary columns

Columns with empty values for most of the rows as well as columns with the same values across 
all rows are dropped in order to have a cleaner dataset. Free form text columns are also dropped 
because we posited that these fields would have more noise and are better tackled at a later stage 
when we have better understanding of the problem.

### 2. Implementing Mice to fill missing values

![image](https://user-images.githubusercontent.com/57429405/125011677-e634ae00-e036-11eb-8968-10cdde483a0d.png)
ANALYZING THE MISSING VALUES

MICE is "multiple imputation by chained equations". Basically, missing data is predicted by 
observed data, using a sequential algorithm that is allowed to proceed to convergence. 
(1) We have started filling in the missing data with plausible guesses at what the values might be. 
(2) for each variable, we have predicted the missing values by modeling the observed values as a 
function of the other variables. 

 imputer = mice.MICEData(df)
 imputer.set_imputer('x1', formula='x2 + np.square(x2) + x3')
 for j in range(20):
   imputer.update_all()
 imputer.data.to_csv("data_after_mice.csv")
 
 At each step, we have updated the predictions of the missing values. Further, we have calculated 
stats model imputation formulas and updated the values available in the specified range. Thus, 
missing values have been successfully filled by implementing Mice.

### 3. Normalize Data with Sklearn.ipynb

![image](https://user-images.githubusercontent.com/57429405/125011702-f5b3f700-e036-11eb-97ca-1e983dea8885.png)
CORRELATION ANALYSIS

We then have approached to perform Normalization on the dataset to bring the data set on the 
same scale. We have split the dataset into 2 data frames out of which 1 needs to be normalized. 
Later, we performed normalization then summarized the transformed data. We have then 
calculated the correlation with interest rate after normalization for performing analysis and 
visualizations.

### Normalizing the data: 
scaler_data = Normalizer().fit(X)
normalizedData = scaler_data.transform(X)

### Summarizing the transformed data
np.set_printoptions(precision=3)

### Displaying transformed data
print(normalizedData[0:5,:])

## 3. FEATURE SELECTION

![image](https://user-images.githubusercontent.com/57429405/125011724-ff3d5f00-e036-11eb-8da9-3cf86c3031a5.png)


Feature engineering is the process of using domain knowledge of the data to create features that 
make machine learning algorithms work. If feature engineering is done correctly, it increases the 
predictive power of machine learning algorithms by creating features from raw data that help 
facilitate the machine learning process.

### Implementing Lasso CV for feature Selection

Lasso (“Least Absolute shrinkage and selection operator”) is a regression analysis method that 
performs both variable selection and regularization in order to enhance the prediction accuracy 
and interpretability of the statistical model it produces. The data is then categorized into three sets 
and treated differently: train, test and cv. We then split the data into dependent and independent 
variables to perform lasso cv for variable selection. Thus, we have successfully implemented 
feature selection by performing dimension reduction using LassoCV. 

### Implementing Feature Tools

![image](https://user-images.githubusercontent.com/57429405/125011758-0cf2e480-e037-11eb-9a56-394b9e9a6698.png)

Feature tool helps the user to find some features that are necessary for mathematical calculation 
like avg, min, max, count etc. Feature tools have a lot of limitations like you need to pass proper 
entities before gaining features out of it. So, if a user needs to find some function related to 
mathematics then these feature tools are useful. Feature tool works well only with the numeric 
data. A user needs to decide manually first what features he needs and then pass the primitives 
accordingly. In short it is manually feature selection.

## MODELS & ALGORITHMS USED:

![image](https://user-images.githubusercontent.com/57429405/125011783-167c4c80-e037-11eb-9e70-b88735822aac.png)


### Linear Regression

Linear regression is a statistical approach for modelling the relationship between a dependent variable with a given set of independent variables
Linear regression technique finds out a linear relationship between x (input) and y (output). It is used to predict a quantitative response Y from the predictor variable X. It is made with an assumption that there's a linear relationship between X and Y
The equation of the above line is :
Y= mx + b
Where b is the intercept and m is the slope of the line

### Random Forests


A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees
The random forest algorithm combines multiple algorithms of the same type i.e. multiple decision trees, resulting in a forest of trees, hence the name "Random Forest". The random forest algorithm can be used for both regression and classification tasks
In case of a regression problem, for a new record, each tree in the forest predicts a value for Y (output). The final value can be calculated by taking the average of all the values predicted by all the trees in forest. Or, in case of a classification problem, each tree in the forest predicts the category to which the new record belongs. Finally, the new record is assigned to the category that wins the majority vote

![image](https://user-images.githubusercontent.com/57429405/125012011-770b8980-e037-11eb-8451-3cf492acccd0.png)



## VISUALIZATION USING TABLEAU AND POWERBI

![image](https://user-images.githubusercontent.com/57429405/125011570-b8e80000-e036-11eb-9bbb-db3ff9402858.png)

![image](https://user-images.githubusercontent.com/57429405/125011602-c604ef00-e036-11eb-95fb-0addf88b4d02.png)








