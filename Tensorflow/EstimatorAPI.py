# https://github.com/sandhyaparna/training-data-analyst/blob/master/courses/machine_learning/deepdive/03_tensorflow/b_estimator.ipynb
# https://googlecoursera.qwiklabs.com/focuses/25433?locale=en

featcols = [
      tf.feature_column_numeric_column("sq_footage"),
      tf.feature_column.categorical_column_with_vocabulary_list("type",["house","apt"]), #type variable has 2 unique values-house, apt
      tf.feature_column_numeric_column("nbeds")
]
model = tf.estimator.LinearRegressor(featcols,'./model_trained') #'./model_trained' is used to create checkpoints in that folder

# featcols can also be - To extract only numeric cols
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
LABEL = CSV_COLUMNS[0]
def feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
  return input_columns
model = tf.estimator.LinearRegressor(feature_columns = feature_cols(), model_dir = OUTDIR)

def train_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df['price'],
    batch_size=128, num_epochs=10, shuffle=True
  )

model.train(train_input_fun(df)) #Trains until 10 epochs starting from checkpoint
model.train(train_input_fun(df), steps=100)  #100 additional steps from checkpoint
model.train(train_input_fun(df), max_steps=100)  #100 steps might be nothing if checkpoint is already there





