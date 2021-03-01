### GCP GitHub 
Repo https://github.com/sandhyaparna/training-data-analyst <br/>

* ML 1.1 Invoking ML APIs: How to open Notebooks, start an instance, clone a repo
  * Github repo https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/CPB100/lab4c/mlapis.ipynb
  * Lab Solution https://www.coursera.org/learn/google-machine-learning/lecture/PVHTl/lab-solution
* ML 1.2 Analyzing data using Datalab (Google Cloud DataLab) and BigQuery: Invoke BigQuery Console, How to use BigQuery within Python Jupyter notebook
  * https://www.coursera.org/learn/google-machine-learning/lecture/Cj3hg/lab-debrief-analyzing-data-using-datalab-and-bigquery   <br/>
* ML 2.1 Improve Data Quality 
  * https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/labs/improve_data_quality.ipynb
  * Solution https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/solutions/improve_data_quality.ipynb
* ML 2.2 Exploratory Data Analysis Using Python and BigQuery
  * https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/solutions/python.BQ_explore_data.ipynb
* ML 2.3 Introduction to Linear Regression
 * https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/solutions/intro_linear_regression.ipynb 
* ML 2.4 Introduction to Logistic Regression
 * https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/launching_into_ml/solutions/intro_logistic_regression.ipynb  



Rent-a-VM to Process Earthquake Data https://www.coursera.org/learn/google-machine-learning/lecture/Qonu0/lab-debrief  <br/>
  Qwiklabs https://www.qwiklabs.com/focuses/1846?catalog_rank=%7B%22rank%22%3A1%2C%22num_filters%22%3A0%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=1953907  <br/>
   <br/>


Activate datalab
* In Google cloud platform, open cloud shell - datalab create vminstancename --zone us-central1-a
* To open instance that has already been created - datalab connect vminstancename

In the datalab clone a github repository: <br/>
%bash <br/>
git clone https://github.com/GoogleCloudPlatform/training-data-analyst <br/>
rm -rf training-data-analyst/.git <br/>



Using Cloud DataPrep - join data, create new features; save data to BigQuery-Perform Aggregation; Use DataLab to visualize data
https://www.qwiklabs.com/focuses/610?locale=en&parent=catalog

Feature Creation https://github.com/sandhyaparna/training-data-analyst/blob/master/courses/machine_learning/feateng/feateng.ipynb <br/>

Tf transform https://github.com/sandhyaparna/training-data-analyst/blob/master/courses/machine_learning/feateng/tftransform.ipynb  <br/>
