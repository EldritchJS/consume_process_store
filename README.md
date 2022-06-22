```
oc new-app eldritchjs/py36centostf:tf~https://github.com/eldritchjs/adversarial_pipeline \
-e KAFKA_BROKERS=kafka:9092 \
-e KAFKA_READ_TOPIC=commands \
-e DBHOST=postgresql \
-e DBNAME=results \
-e DBUSERNAME=redhat \
-e DBPASSWORD=redhat \
--name consume_process_store
```
