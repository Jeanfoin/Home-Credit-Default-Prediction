# **Home Credit Default Capstone Project**

## **What is Home Credit?**

Home Credit is an international non-bank financial institution founded in 1997 in the Czech Republic. It operates in nine countries, including the United States, and specializes in installment lending (cash loans and revolving loans), primarily targeting individuals with little or no credit history. In 2018, Home Credit launched a Kaggle competition with the goal of building the best possible model to predict whether a loan applicant will default on their loan.

## **What is to be done?**

The first order of business is to thoroughly understand the data at hand by conducting an extensive Exploratory Data Analysis (EDA) of each dataset provided.

Once this analysis is complete, the focus will shift to building the most accurate classifier possible, along with its preprocessing pipeline. This will involve experimenting with several algorithms and various preprocessing steps, such as encoding and imputation. To achieve this, we have access not only to the loan applicant's current application but also to their entire loan history with Home Credit. For U.S. citizens, we also have access to their previous loan history with Credit Bureau-referenced lending companies.

Finally, once the model is built, it will be deployed so that a Home Credit employee can access it via a web browser. The employee will be able to enter the loan applicant's data and receive an estimation of whether the loan should be approved or rejected.

Why is it so important to accurately detect whether a loan applicant is likely to default? Each defaulted loan represents a financial loss for the lending institution. A precise machine learning model that can efficiently predict the likelihood of default would be of immense value to any lender. It would not only help them avoid losses from defaulters but also enable them to extend credit to applicants whose situations might be challenging for an expert to assess.

## **The Data**

As mentioned earlier, the data consists of six different dataframes, all related by two integer keys: `SK_ID_CURR`, which corresponds to the current application, and `SK_ID_PREVIOUS`, which identifies any previous loans or applications. The original data can be found [here](https://www.kaggle.com/competitions/home-credit-default-risk/overview).

<div style="display: flex; align-items: center;">   <img src="./figures/data_scheme.png" alt="drawing" width="800" style="margin-right: 20px;"/>   <p style="max-width: 400px;">     <strong>Scheme of the data organization</strong>:  applicant is the main dataset, all others will often be referred as additional   </p> </div>

Given the substantial amount of data, processing it as is may lead to challenges such as memory limitations and increased computation time. To address this, we'll begin by tackling these issues. The `dev_file_creation` notebook loads each dataset and optimizes memory usage by downcasting the data types of each feature (e.g., converting `object` to `category`, `int64` to `int32`, etc.) where possible.

Next, the notebook splits the main dataset into training and test sets (stored in `/app_data`) and extracts a sample of the training set, referred to as the sample set (stored in `/sample_data`). For the additional datasets, it selects a sample of each one based on the `SK_ID_CURR` values from the sample of the main dataframe and stores these in `/sample_data`. Unlike the main dataset, these additional datasets are not split into training and test sets; instead, their memory-optimized versions are stored in the `/add_data` directory. These additional dataframes can be considered as a remote database for reference. I've decided to use the `.parquet` extension for the memory-reduced datasets because it efficiently stores data by columns, reducing storage space. Parquet also preserves the data types of each dataframe, ensuring accuracy when the data is loaded back into memory.


## How to run this project

Each notebook is self-contained and can be run independently of the others. However, before proceeding, you need to have the data. To get started, download the [original data](https://www.kaggle.com/competitions/home-credit-default-risk/overview) and place it in a directory named `original_data` at the root of your project. Once this is done,  you'll need to install all the necessary libraries and dependencies that are listed in `requirements.txt` by running:

```pip install -r requirements.txt```

 If you are working in a new virtual environment you will need to install jupyter and set the new kernel

```pip install jupyter```

 ```pip install ipykernel```

```python3 -m ipykernel install --user --name=kernel_name```

 Then, simply run the `dev_files_creation` notebook, which will automatically create all the necessary datasets. 

To avoid long computation times in certain notebooks, such as those involving Recursive Feature Selection and Bayesian Optimization for hyperparameters tuning, I’ve used cell magic to store the results. 

## Structure of this project

This project is structured as follows:

- **/notebooks directory:** This directory contains the nine final notebooks produced during the project. Each notebook can be run independently and in any order. Here’s a brief description of each:
  - **dev_files_creation.ipynb:** The primary purpose of this notebook is to reduce memory usage by downcasting the data types of each dataset where possible, and then saving them as `.parquet` files.
  - **EDA notebooks:** Notebooks with names starting with "EDA" and ending with a dataset name contain the Exploratory Data Analysis for that specific dataset. The EDAs for the additional datasets follow a similar approach: data is loaded and inspected (checking for missing values, anomalies, etc.), functions are created to extract features for the future classifier, and sometimes features are aggregated over `SK_ID_PREV` before further aggregation over `SK_ID_CURR`. The resulting features are then analyzed using Kendall correlation heatmaps. While the features in these notebooks may differ from the final ones, the process demonstrates the overall approach.
  - **EDA_sample_application.ipynb:** This notebook contains the EDA of the main dataset. It stands out because a more thorough analysis was possible, using a wide range of relevant statistical tools (correlation, hypothesis testing, etc.) to achieve a comprehensive and unbiased description. Part of this notebook is dedicated to filtering the dataset to reduce the number of features, and the final section focuses on training different models on the main dataset alone to evaluate performance.
  - **Model_and_features_selection.ipynb:** This notebook outlines how I merged all datasets and addressed data preprocessing. It also covers model selection by comparing the performance of different classifiers and preprocessing steps. After choosing the LightGBM classifier, I used Recursive Feature Selection for feature selection. By the end of this notebook, I had a clear understanding of which features should be used for the task.
  - **Final_model.ipynb:** This notebook contains the final model that will be deployed. I used Bayesian optimization to tune the hyperparameters, and the functions for the final model were optimized and rewritten as custom classes to simplify deployment, especially since ColumnTransformer objects can be challenging to serialize.
- **/utilities directory:** This directory contains various utility files with custom functions that were frequently used throughout the project.
  - **stat_utilities.py:** Contains functions for handling statistical operations, such as finding missing values (`get_missing_value`), calculating the entropy of categorical features (`calculate_cat_entropy`), and performing hypothesis testing (`corr_cramersV`).
  - **plot_utilities.py:** Includes several plotting utility functions to create and customize graphs.
  - **features_creation.py:** A draft version of functions used to create features from the additional datasets. These functions will be updated and refined later in the project after selecting the final model for deployment.
  - **final_features.py:** Contains the final versions of preprocessing functions to be used in the deployed model. Each preprocessing step has been encapsulated in a custom class to streamline the deployment process.
- **/pkl:** This directory contains three serialized objects, saved using `joblib`:
  - **fsp_fitted.joblib:** The raw model after feature selection was completed in the `Model_and_features_selection.ipynb` notebook.
  - **lgbm_best.joblib:** The final pipeline, which represents the model that will be deployed.
  - **shap_explainer.joblib:** A serialized SHAP object used in the Flask application where the model is deployed, helping to generate SHAP waterfall plots to explain individual predictions.
- **/figures:** This directory contains the figures embedded in this README file.
- **requirements.txt:** text file listing all the necessary libraries to run the notebook. You can install them using the command: ```pip install -r requirements.txt```
- **/app_data:** Contains the Parquet versions of the main dataset (application) split into training and test sets. (after running dev_file_creation)
- **/add_data:** Contains the Parquet versions of all additional datasets. (after running dev_file_creation)
- **/sample_data:** Contains the Parquet versions of the sampled datasets used for Exploratory Data Analysis (EDA). (after running dev_file_creation)


## **My approach**

For this project, I decided that the most effective approach was to compartmentalize the study of each dataset in separate notebooks before integrating them. This is why you’ll find a different EDA for each notebook, even though they share a very similar structure. Additionally, I made it a priority to use tools from `sklearn` wherever possible. I specifically designed the project so that my model could be integrated into an `sklearn` Pipeline, which helps prevent any potential data leakage. I intended this pipeline to serve as a proof of concept for how I envision such an application being used in a professional environment. In my vision, the datasets containing an applicant's loan history would function as a remote database. This way, whenever an applicant applies for a new loan, their past history would be automatically retrieved, and additional features would be computed on the fly. I believe that this approach would make the system easier to update, upgrade, and maintain. The process is illustrated in the scheme shown below. 

<div style="display: flex; align-items: center;">   <img src="./figures/pipeline_scheme.png" alt="drawing" width="800" style="margin-right: 20px;"/>   <p style="max-width: 400px;">     <strong>Scheme of the Final Pipeline</strong>: In the deployed model, the distant databases are represented by additional datasets stored in the <code>/add_data</code> directory.   </p> </div>

## **Results**

* **Thorough EDAs** were conducted on every dataset to detect anomalies and missing values and to build diagnostics where possible. Following this, the process of feature creation was exemplified, resulting in an initial batch of over 500 features. Many features were computed across different time windows (`EVER`, `LAST_YEAR`, `LAST_3_MONTHS`) to capture the temporality in a loan applicant's repayment behavior. These would later be refined through recursive selection to reduce their number. One key observation from the EDAs was the strong relationship between the `EXT_SOURCE` features and the target variable. These scores, provided by an external source, are intended to evaluate the reliability of loan applicants. Unfortunately, a significant portion of these values was missing.

  <div style="display: flex; align-items: center;">   <img src="./figures/target_variable.png" alt="drawing" width="800" style="margin-right: 20px;"/>   <p style="max-width: 400px;">     <strong>Target variable distribution:</strong> We see a clear imbalance, the percentage of defaulters being slightly above 8%.   </p> </div>

* **Objective 1: Classifier Development**
  The primary goal of this project was to build a classifier capable of predicting whether a loan applicant would default on their loan. The dataset was highly imbalanced, with only 8% of applicants being defaulters, making it challenging to create an effective model. Instead of employing sampling strategies, I opted for classifier-specific methods to address this imbalance by adjusting class weights. This decision was informed by previous experiences where sampling methods like undersampling, random oversampling, and SMOTE had not proven very effective.

  A preliminary version of the model, trained solely on the main dataset (`EDA_sample_application.ipynb`), achieved a ROC-AUC score of approximately **0.74**. After incorporating features from additional datasets and tuning the hyperparameters, the model’s performance improved to a ROC-AUC score of **0.769**. While this improvement might seem marginal, it could represent millions of dollars in avoided losses for Home Credit, highlighting the value of even small gains in predictive accuracy.

  <div style="display: flex; align-items: center;">   <img src="./figures/classifier_result.png" alt="drawing" width="1100" style="margin-right: 20px;"/>   <p style="max-width: 400px;">     <strong>Final model performance:</strong> As we can see on the confusion matrix, our model makes a lot of false positive.   </p> </div>
  
* **Objective 2: Understanding the Model** Among the top 20 most important features, there are notable differences between the feature importance rankings provided by SHAP and LightGBM. For example, while SHAP ranks gender among the top 3 most important features, it doesn't even appear in LightGBM's top 20. However, both methods consistently highlight the significance of the `CREDIT_ANNUITY_RATIO` and `EXT_SOURCE`-related features.
  
  Interestingly, none of the features engineered from the additional datasets appear in either the LightGBM or SHAP rankings. This suggests that the information in the loan application form alone captures most of the crucial details needed to predict whether a loan applicant is likely to default.
  
  <div style="display: flex; align-items: center;">   <img src="./figures/feature_importances.png" alt="drawing" width="1200" style="margin-right: 20px;"/>   <p style="max-width: 400px;">     <strong>Features importances:</strong> From LightGBM on the left, and SHAP on the right.   </p> </div>
  
* **Objective 3: Business Application of the Classifier**
  Beyond just developing a classifier, I explored how it could be applied in a business context. While it's unlikely that we can achieve a perfect score for this task (even the best models in the Kaggle competition scored around 0.8), companies like Home Credit are more concerned with reducing losses and increasing profits. One way to achieve this is by tuning the model's decision threshold to align with business objectives.

  I experimented with a simple model, which suggested that Home Credit should accept all applicants to maximize profit. Of course, this is not feasible for various reasons. However, with additional data (e.g., recovery rates, installment schedules) and deeper domain knowledge, a more sophisticated model could be developed. This enhanced model could not only maximize profits but also tailor repayment schemes to individual applicants.

* **Objective 4: Deployment**
  The final objective was to make the model deployable, which I achieved by developing a robust pipeline. The final model has been deployed on Google Cloud Platform and is accessible at [this address](https://cz-turing-hcd-app-305869428789.us-central1.run.app/). The application starts with a form mimicking a loan application. Don't be alarmed by the number of features; only a few are required: `SK_ID_CURR`, `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, and `DAYS_BIRTH`. The embedded Flask app handles everything else. After submitting the form, you’ll receive a decision on whether the application is accepted or rejected, along with a SHAP waterfall plot providing insights into the decision. Below are screenshots demonstrating these steps.

<div style="display: flex; align-items: center;">   <img src="./figures/form_screen.png" alt="drawing" width="800" style="margin-right: 20px;"/>   <p style="max-width: 400px;">     <strong>Screenshot of the deployed model initial form</strong> </p> </div>

<div style="display: flex; align-items: center;">   <img src="./figures/result_screen.png" alt="drawing" width="800" style="margin-right: 20px;"/>   <p style="max-width: 400px;">     <strong>Screenshot of the deployed model result</strong> </p> </div>


## **Opportunities for Improvement and Future Directions**

There is obviously much room for improvement, which highlights the importance of focusing on the following next steps:

- **Improve Our Classifier:** The first priority should be to enhance our classifier, aiming to get closer to the top scores on the Kaggle leaderboard. Achieving this will likely require significant computational resources.
- **Better Imputation of Key Features:** Improving the imputation of critical features, such as `EXT_SOURCE`, could greatly enhance model performance. This, in turn, would support the goal of improving the classifier. Additionally, exploring the creation of new feature scores might further boost accuracy.
- **Experiment with Neural Networks:** Neural networks could offer a promising alternative to traditional machine learning classifiers. It would be worthwhile to experiment with training neural networks on this dataset to evaluate their effectiveness.
