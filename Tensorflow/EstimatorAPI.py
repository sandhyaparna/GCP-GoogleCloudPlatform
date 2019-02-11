# https://github.com/sandhyaparna/training-data-analyst/blob/master/courses/machine_learning/deepdive/03_tensorflow/b_estimator.ipynb
# https://googlecoursera.qwiklabs.com/focuses/25433?locale=en

### Columns
# 1)
featcols = [
      tf.feature_column_numeric_column("sq_footage"),
      tf.feature_column.categorical_column_with_vocabulary_list("type",["house","apt"]), #type variable has 2 unique values-house, apt
      tf.feature_column_numeric_column("nbeds")
]
model = tf.estimator.LinearRegressor(featcols,'./model_trained') #'./model_trained' is used to create checkpoints in that folder

# 2) featcols can also be - To extract only numeric cols
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
LABEL = CSV_COLUMNS[0]
def feature_cols():
  input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
  return input_columns
# Linear Regression with tf.Estimator framework
tf.logging.set_verbosity(tf.logging.INFO)
OUTDIR = 'taxi_trained'
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
model = tf.estimator.LinearRegressor(feature_columns = feature_cols(), model_dir = OUTDIR)

### Trains set
# 1)
def train_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df['price'],
    batch_size=128, num_epochs=10, shuffle=True
  )
model.train(train_input_fun(df)) #Trains until 10 epochs starting from checkpoint
model.train(train_input_fun(df), steps=100)  #100 additional steps from checkpoint
model.train(train_input_fun(df), max_steps=100)  #100 steps might be nothing if checkpoint is already there
# 2)
model.train(input_fn = make_train_input_fn(df_train, num_epochs = 10))

### Then evaluate on Validation dataset
### Later predict on Test data

