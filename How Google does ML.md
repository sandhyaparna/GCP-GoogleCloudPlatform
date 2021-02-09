
* Created over 4000 Tensorflow ML models
* Infrastructure for ML - How do you process batch data and stream data the same way? - On GCP, the key services are all serverless and they're all managed infrastructure.
  * Dataflow process batch and stream data in the same way
* Data is key to any ML model. Invest in getting more data
* A ML model is a mathematical function. Models generalize from the examples provided to it
* The key to making a machine learning model generalized is data, and lots and lots of it. Having labeled the data is a precondition for successful machine learning.
* Prediction is nothing but inference
* BigQuery - SQL code
* Scaling out - As your data size increases, batching and distribution become extremely important. You will need to take your data and split it into batches, and then you need to train, but then you need to also distribute your training over many machines. Parameter servers that form a shared memory that's updated during each epoch are needed. 
* A single business problem migh be a combination of multiple ML models. For eg - To forecast whether an item will go out of stock. You may have to break this problem down into smaller problems, based on your knowledge of the business. For example, your first model might be to the predicted demand for the product at the store location. Your second model might predict the inventory of this item at your suppliers warehouse and at nearby stores. You might need a third model to predict how long it's going to take them to stock your product and use this to predict which supplier you will ask to refill the shelf and when. Of course, all these models themselves might be more complex. The model to predict the demand for milk is going to be very different from the model to predict the demand for dry noodles. The model for restocking electronics is very different from the model for restocking furniture. There is not one ML model. There are dozens of ML models per product.









