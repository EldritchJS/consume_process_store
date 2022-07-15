from os import environ
from kafka import KafkaConsumer
import argparse
import logging
import os
import time
from datetime import datetime
from json import loads
from urllib import request
from zipfile import ZipFile
from porn_det import prnDet

def scanFiles(basedir, threshold=0.7):
    # walk the directory tree
    # for each file
    # results = prnDet(filename)
    # if results[0] == 1 and results[1] > threshold:
    # delete? rename? 
    return 0

def main(args):
    logging.info('brokers={}'.format(args.brokers))
    logging.info('topic={}'.format(args.topic))
    logging.info('creating kafka consumer')   

    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.brokers,
        value_deserializer=lambda val: loads(val.decode('utf-8')))
    logging.info("finished creating kafka consumer")

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
                os.remove('./data/batch.zip')
                if(message.value['scan']):
                    scanFiles('./data')
                    

def get_arg(env, default):
    return os.getenv(env) if os.getenv(env, "") != "" else default

def parse_args(parser):
    args = parser.parse_args()
    args.brokers = get_arg('KAFKA_BROKERS', args.brokers)
    args.topic = get_arg('KAFKA_TOPIC', args.topic)
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
            help='Topic to read from, env variable KAFKA_TOPIC',
            default='commands')
    cmdline_args = parse_args(parser)
    main(cmdline_args)
    logging.info('exiting')


