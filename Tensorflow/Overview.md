Tensorflow Website https://www.tensorflow.org/

TensorFlow is an open source, high performance, library for numerical computation.  <br/>
Tensordlow is a lazy evaluation model - write a DAG and then you run the DAG in the context of a session to get results
But in tf.eager the evaluation is immediate and it's not lazy but it is typically not used in production programs and used only for development. <br/>
TensorBoard is used to visulaize tensorfloe graphs <br/>
A variable is a tensor whose value is initialized and then the value gets changed as a program runs. <br/>
  
#### Tensorflow APIs
* Core Tensorflow Python API - Numeric processing code, add, subtract, divide, matrix multiply etc. creating variables, creating tensors, getting the shape, all the dimensions of a tensor, all that core basic numeric processing stuff.  <br/>
* Components useful when building custon NN models <br/>
tf.layers - a way to create a new layer of hidden neurons, with a ReLU activation function. <br/>
tf.losses - a way to compute cross entropy with Logits.  <br/>
tf.metrics - a way to compute the root mean square error and data as it comes in. <br/>
* tf.estimator - knows how to do distributed training, it knows how to evaluate, how to create a checkpoint, how to Save a model, how to set it up for serving. It comes with everything done in a sensible way, that fits most machine learning models in production. <br/>
 
#### Estimator API
* Quick model -  Many standard pre-made estimator models
* Checkpoints to pause and resume your training
* Out-of-memory datasets - Estimators are designed with a data set API that handles out of memory data sets.  
* Train/eval/monitor - You can not train a large network without seeing how its doing. Estimators automatically surface key metrics during training that you can visualize in Tensor board.
* Distributed Training - Estimators come with the necessary cluster execution code already built in. 
* Hyper-parameter tuning of ML-engine
* Production:serving predictions from a trained model

#### How to change optimizer, learning rate, batch size
* Batch size is controlled in input function
* Learning rate, regularization are parameters of optimizer algorithm and pass it to the estimator 
* Steps = (Number of epochs * Number of Samples) / Batch Size. Tensorflow doesn't know epoch, so steps should be used
* If you decrease learning rate, you'll have to train for more epochs.

#### Hyper parameter tuning - Google Vizier - Cloud ML Engine
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf <br/>
* Express the hyperparameters tuning in need of tuning as a command-line argument - parser.add_argument
* Ensure differnt iterations of different training trails dont clobber each other - Naming Convention - Use some word as a suffix 
* Supply those hyperparameters to the training job











