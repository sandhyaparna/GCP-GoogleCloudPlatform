# Extract, Train, Evaluate & Test using Tensorflow
# https://github.com/sandhyaparna/training-data-analyst/blob/master/courses/machine_learning/deepdive/03_tensorflow/b_estimator.ipynb
# https://googlecoursera.qwiklabs.com/focuses/25433?locale=en

### 
featcols = [
      tf.feature_column_numeric_column("sq_footage"),
      tf.feature_column.categorical_column_with_vocabulary_list("type",["house","apt"]), #type variable has 2 unique values-house, apt
      tf.feature_column_numeric_column("nbeds")  ]
# featcols[0] in this example implies u want to bucketize sq_footage column into the specified buckets 
# sq_footage is fed into the model as both numeric and category or u can also replace
featcols.append(
      fc.buketized_column(featcols[0],[500,1000,2000]))
# Embeddings in tensorflow 
# For latitude & longitude
latbuckets = np.linespace(38,42,nbuckets).tolist() #Define intervals into which we want lat to be discretized into. nbuckets for equally spaced intervals
lonbuckets = np.linespace(-76,-72, nbuckets).tolist() #Define intervals into which we want lon to be discretized into
b_lat = fc.bucketized_column(house_lat, latbuckets)
b_lon = fc.bucketized_column(house_lon, lonbuckets)
loc = fc.crossed_Column([b_lat,b_lon], nbuckets*nbuckets) #Creating feature crosses from lat & lon
eloc = fc.embedding_column(loc,nbuckets//4) # Create nbuckets by 4 embeddings
#
model = tf.estimator.LinearRegressor(featcols,'./model_trained') #'./model_trained' is used to create checkpoints in that folder

def train_input_fn(df):
  return tf.estimator.inputs.pandas_input_fn(
    x = df,
    y = df['price'],
    batch_size=128, 
    num_epochs=10,
    shuffle=True)
model.train(train_input_fun(df)) #Trains until 10 epochs starting from checkpoint
model.train(train_input_fun(df), steps=100)  #100 additional steps from checkpoint
model.train(train_input_fun(df), max_steps=100)  #100 steps might be nothing if checkpoint is already there
# Then evaluate on Validation dataset
# Later predict on Test data

### Distributed Training  - shuffling is more imp
run_config = tf.estimator.RunConfig(model_dir=output_dir, save_summary_steps=100, save_checkpoints_steps=2000)
estimator = tf.estimator.LinearRegressor(feature_columns=featcolls, config=run_config)
# Train
train_Spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=5000)
# export
export_latest = tf.estimator.LatestExporter(serving_input_receiver_fn=serving_input_fn)
# Eval
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=100, throttle_secs=600, exporters=...) #Steps is eval on 100 batches, throttle_secs-eval no more than every 10mins
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

### Dataset API
# text line dataset to load data from a csv file
def decode_line(row):
    cols = tf.decode_csv(row, record_defaults = [[0],['house'],[0]])
    features = {{'sq_footage':cols[0], 'type':cols[1]}
    labels = cols[2]
  return features, label
# Dataset 
# Normal dataset
dataset = tf.data.TextLineDataset("train_1.csv") \
                 .map(decode_line)
# Large datsets
dataset = tf.data.Dataset.list_files("train.csv-*")  \
                 .flat_map(tf.data.TextLineDataset) \
                 .map(decode_line
# Shuffles data, repeats into 15 epochs and seperate into batches
dataset = dataset.shuffle(1000) \
                 .repeat(15)    \
                 .batch(128)                
# Data is loaded progresively - They return TensorFlow nodes, and these nodes return data when they get executed
# Receives data from input nodes, features & labels. These nodes iterate on the data set and return one batch of data every time that they get executed in the training loop.
def input_fn():
  features, label = dataset.make_one_shot_iterator().get_next()
  return features, label         
# Launches training loop 
model.train(input_fn)  

                      
            

