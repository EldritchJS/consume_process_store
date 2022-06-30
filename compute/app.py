import psycopg2
from psycopg2.extras import RealDictCursor
from os import environ
from kafka import KafkaConsumer
import argparse
import logging
import os
import time
from datetime import datetime
from json import loads
from urllib import request
import shutil
from zipfile import ZipFile

def main(args):
    logging.info('brokers={}'.format(args.brokers))
    logging.info('topic={}'.format(args.topic))
    logging.info('creating kafka consumer')   

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.brokers,
        value_deserializer=lambda val: loads(val.decode('utf-8')))
    logging.info("finished creating kafka consumer")

    conn = psycopg2.connect(
        host = args.dbhost,
        port = 5432,
        dbname = args.dbname,
        user = args.dbusername,
        password = args.dbpassword)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    logging.info('Creating table if not exists')
    query = 'CREATE TABLE IF NOT EXISTS results (ID varchar(40) NOT NULL, CLUSTERS integer, SCORE integer)'
    cur.execute(query)
    cur.close()
    conn.commit()
    conn.close()

    while True:
        for message in consumer:
            logging.info('Received {}'.format(message.value))
            # Parse the message
            if (message.value['command'] == 'Download') and (message.value['url']):
                logging.info('Received {} command with {} location'.format(message.value['command'],message.value['url']))
                logging.info('Download the data here')
                os.mkdir("./data")
                request.urlretrieve(message.value['url'], filename="./data/batch.zip")
                with ZipFile('./data/batch.zip', 'r') as zipObj:
                    zipObj.extractall(path='./data')
                # Process the data
                logging.info('TODO: Process the data here')
                shutil.rmtree('./data')
                # Store the results
                logging.info('Store results here')
                try:
                    logging.info('Connecting to DB')
                    conn = psycopg2.connect(
                        host=args.dbhost,
                        port=5432,
                        dbname=args.dbname,
                        user=args.dbusername,
                        password=args.dbpassword)
                    logging.info('Connected, creating cursor')
                    cur = conn.cursor(cursor_factory=RealDictCursor)
                    query = 'INSERT INTO results(ID, CLUSTERS, SCORE) VALUES (%s,13,19)'
                    logging.info('Cursor created, sending query {}'.format(query))
                    cur.execute(query, (datetime.now().strftime('%m/%d/%Y-%H:%M:%S'),))
                    logging.info('Closing cursor')
                    cur.close()
                    logging.info('Committing change')
                    conn.commit()
                    logging.info('Closing connection')
                    conn.close()
                except Exception:
                    logging.info('Got exception with postgresql')
                    continue 
                time.sleep(0.3) # Artificial delay for testing

def get_arg(env, default):
    return os.getenv(env) if os.getenv(env, "") != "" else default

def parse_args(parser):
    args = parser.parse_args()
    args.brokers = get_arg('KAFKA_BROKERS', args.brokers)
    args.topic = get_arg('KAFKA_TOPIC', args.topic)
    args.dbhost = get_arg('DBHOST', args.dbhost)
    args.dbname = get_arg('DBNAME', args.dbname)
    args.dbusername = get_arg('DBUSERNAME', args.dbusername)
    args.dbpassword = get_arg('DBPASSWORD', args.dbpassword)
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--brokers',
            help='The bootstrap servers, env variable KAFKA_BROKERS',
            default='kafka:9092')
    parser.add_argument(
            '--topic',
            help='Topic to write to, env variable KAFKA_TOPIC',
            default='commands')
    parser.add_argument(
            '--dbhost',
            help='hostname for postgresql database, env variable DBHOST',
            default='postgresql')
    parser.add_argument(
            '--dbname',
            help='database name to setup and watch, env variable DBNAME',
            default='resultsdb')
    parser.add_argument(
            '--dbusername',
            help='username for the database, env variable DBUSERNAME',
            default='redhat')
    parser.add_argument(
            '--dbpassword',
            help='password for the database, env variable DBPASSWORD',
            default='redhat')
    cmdline_args = parse_args(parser)
    main(cmdline_args)
    logging.info('exiting')


