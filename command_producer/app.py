from os import environ
import argparse
import logging
import os
import time
from kafka import KafkaProducer
from json import dumps

from compute import red_hat_black_box

def main(args):
    logging.info('brokers={}'.format(args.brokers))
    logging.info('topic={}'.format(args.topic))
    logging.info('creating kafka producer')

    producer = KafkaProducer(bootstrap_servers=args.brokers,
                             value_serializer=lambda x: 
                             dumps(x).encode('utf-8'))

    logging.info("finished creating kafka producer")

    while True:
        for i in range(13):
            url_string = 'dummyurl' + str(i)
            message_dict = {'command':'Download', 'url':url_string}
            logging.info('Sending message {}'.format(str(i)))
            producer.send(args.topic, value=message_dict)
            time.sleep(int(args.latency))

def get_arg(env, default):
    return os.getenv(env) if os.getenv(env, "") != "" else default

def parse_args(parser):
    args = parser.parse_args()
    args.brokers = get_arg('KAFKA_BROKERS', args.brokers)
    args.topic = get_arg('KAFKA_TOPIC', args.topic)
    args.latency = get_arg('LATENCY', args.latency)
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--latency',
            help='Time between producing messages, env variable LATENCY',
            default='5')
    parser.add_argument(
            '--brokers',
            help='The bootstrap servers, env variable KAFKA_BROKERS',
            default='kafka:9092')
    parser.add_argument(
            '--topic',
            help='Topic to write to, env variable KAFKA_TOPIC',
            default='commands')
    cmdline_args = parse_args(parser)
    main(cmdline_args)
    logging.info('exiting')


