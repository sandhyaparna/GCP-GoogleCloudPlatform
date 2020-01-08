### Streaming data - Kaggle - Restaurant in Phoenix - data related to checkins
* Kaggle  https://www.kaggle.com/yelp-dataset/yelp-dataset

* Checkin API, Reviews API
* Data flow pipelines using SQL

### CloudPub/Sub - Used to ingest data in realtime - Streaming ingestion


### Cloud DataFlow

Introduction to GCP and Apache Beam
Google Cloud Platform provides a bunch of really useful tools for big data processing. Some of the tools I will be using include:
Pub/Sub is a messaging service that uses a Publisher-Subscriber model allowing us to ingest data in real-time.
DataFlow is a service that simplifies creating data pipelines and automatically handles things like scaling up the infrastructure which means we can just concentrate on writing the code for our pipeline.
BigQuery is a cloud data warehouse. If you are familiar with other SQL style databases then BigQuery should be pretty straightforward.
Finally, we will be using Apache Beam and in particular, we will focus on the Python version to create our pipeline. This tool will allow us to create a pipeline for streaming or batch processing that integrates with GCP. It is particularly useful for parallel processing and is suited to Extract, Transform, and Load (ETL) type tasks so if we need to move data from one place to another while performing transformations or calculations Beam is a good choice.








