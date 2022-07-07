This assumes the Kafka operator is installed on your OpenShift cluster. If so, change the contents of `kafka-cluster.yaml` and `kafka-topic.yaml` to suit your application. Typically it's advisable to at least put a pre or postfix to the kafka broker and database application names. 

Use the OpenShift webui to instantiate a Postgresql ephemeral instance and note your chosen database host name, your database name, database user and password as you'll need them in later steps.


Then get Kafka running via the following (Note: the Kafka user operator should also be used in a live/secure system):

Editing yaml notes: 

kafka-cluster.yaml should have its metadata --> name field changed to <YOUR_PREFIX>-cluster 
kafka-topic.yaml should have its metadata --> name field changed to your desired topic name and metadata --> labels --> strimzi.io/cluster field changed to <YOUR_PREFIX>-cluster

```
oc create -f kafka-cluster.yaml
oc create -f kafka-topic.yaml
```

Now you can start an instance of the consume/compute application.


```
oc new-app https://github.com/eldritchjs/consume_process_store#<YOUR_BRANCH> \
--context-dir=compute \
-e KAFKA_BROKERS=<YOUR_PREFIX>-cluster-kafka-brokers:9092 \
-e KAFKA_TOPIC=<YOUR_KAFKA_TOPIC> \
-e DBHOST=<YOUR_PREFIX>-postgresql \
-e DBNAME=<YOUR_DBNAME> \
-e DBUSERNAME=<YOUR_DBUSERNAME> \
-e DBPASSWORD=<YOUR_DBPASSWORD> \
--strategy=docker \
--name <YOUR_PREFIX>-compute
```


You can start a basic webapp with commands and results endpoints to send messages onto Kafka and retrieve Postgresql results:


```
oc new-app openshift/python:latest~https://github.com/eldritchjs/consume_process_store#<YOUR_BRANCH> \
--context-dir=webapp \
-e KAFKA_BROKERS=<YOUR_PREFIX>-cluster-kafka-brokers:9092 \
-e KAFKA_TOPIC=<YOUR_KAFKA_TOPIC> \
-e DBHOST=<YOUR_PREFIX>-postgresql \
-e DBNAME=<YOUR_DBNAME> \
-e DBUSERNAME=<YOUR_DBUSERNAME> \
-e DBPASSWORD=<YOUR_DBPASSWORD> \
--name <YOUR_PREFIX>-webapp
```

Be sure to expose your webapp as a service so you can access it. 

```
oc expose svc/<YOUR_PREFIX>-webapp
```

To send some commands over the topic (for testing the Kafka and Postgresql setup only):


```
oc new-app openshift/python:latest~https://github.com/eldritchjs/consume_process_store \
  --context-dir=command_producer \
  -e KAFKA_BROKERS=<YOUR_PREFIX>-cluster-kafka-brokers:9092 \
  -e KAFKA_TOPIC=<YOUR_KAFKA_TOPIC> \
  --name <YOUR_PREFIX>-command-producer
```

