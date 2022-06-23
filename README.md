```
oc create -f kafka-cluster.yaml
oc create -f kafka-topic.yaml
```

```
oc new-app openshift/python:latest~https://github.com/eldritchjs/consume_process_store \
-e KAFKA_BROKERS=eldritchjs-cluster-kafka-brokers:9092 \
-e KAFKA_TOPIC=commands \
-e DBHOST=postgresql \
-e DBNAME=results \
-e DBUSERNAME=redhat \
-e DBPASSWORD=redhat \
--name consume-process-store
```


To send some commands over the topic:

```
oc new-app openshift/python:latest~https://github.com/eldritchjs/consume_process_store \
  --context-dir=command_producer \
  -e KAFKA_BROKERS=eldritchjs-cluster-kafka-brokers:9092 \
  -e KAFKA_TOPIC=commands \
  --name command-producer
```