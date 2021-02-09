
* Created over 4000 Tensorflow ML models
* Infrastructure for ML - How do you process batch data and stream data the same way? - On GCP, the key services are all serverless and they're all managed infrastructure.
  * Dataflow process batch and stream data in the same way
* Data is key to any ML model. Invest in getting more data
* A ML model is a mathematical function. Models generalize from the examples provided to it
* The key to making a machine learning model generalized is data, and lots and lots of it. Having labeled the data is a precondition for successful machine learning.
* Prediction is nothing but inference
* BigQuery - SQL code
* Scaling out - As your data size increases, batching and distribution become extremely important. You will need to take your data and split it into batches, and then you need to train, but then you need to also distribute your training over many machines. Parameter servers that form a shared memory that's updated during each epoch are needed. 

#### Examples of ML models
* A single business problem migh be a combination of multiple ML models. For eg - To forecast whether an item will go out of stock. You may have to break this problem down into smaller problems, based on your knowledge of the business. For example, your first model might be to the predicted demand for the product at the store location. Your second model might predict the inventory of this item at your suppliers warehouse and at nearby stores. You might need a third model to predict how long it's going to take them to stock your product and use this to predict which supplier you will ask to refill the shelf and when. Of course, all these models themselves might be more complex. The model to predict the demand for milk is going to be very different from the model to predict the demand for dry noodles. The model for restocking electronics is very different from the model for restocking furniture. There is not one ML model. There are dozens of ML models per product.
* The Google translate app let's you point a phone camera at your street sign and it translates a sign for you. This is a good example of a combination of several models that is quite intuitive. One model to find the sign, another model to read the sign, to do optical character recognition on it. A third model to translate the sign, or maybe a third model to detect the language and a fourth model to translate the sign. And a fifth model to superimpose the translated text. Perhaps even sixth model to select the font to use. 
* Gmail smart reply generates 3 different replies. seq to seq model
* RankBrain - Deep NN for search ranking
  * Meaning of your query: Understanding intent is fundamentally about understanding language, and is a critical aspect of Search.language models to try to decipher what strings of words we should look up in the index: Spelling mistakes, synonym search, category of info, langugae of the text, fresh content for latest scores etc. Word "change" meaning is different in different sentences - How to change (replace) a light bulb; Does post office chnage (exchange) foreign currency; How to change (adjust) brightness on a laptop
  * Relevance of webpages: Next, algorithms analyze the content of webpages to assess whether the page contains information that might be relevant to what you are looking for. If key words in the serach query are part of the body or content of the web page
  * Quality of content: Importance is given to reliable sources
  * Usability of webpages: Ease of usage - Is website designed for all device types and sizes; different browsers, page loading time
  * Context and settings: Information such as your location, past Search history and Search settings all help us to tailor your results to what is most useful and relevant for you in that moment.
  
  
  








