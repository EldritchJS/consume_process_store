import os
import logging
import argparse
from flask import Flask
import psycopg2
from psycopg2.extras import RealDictCursor
from kafka import KafkaProducer
from json import dumps


cmdline_args = []
app = Flask(__name__)


@app.route('/results')
def results():
    conn = psycopg2.connect(
        host=cmdline_args.dbhost,
        port=5432,
        dbname=cmdline_args.dbname,
        user=cmdline_args.dbusername,
        password=cmdline_args.dbpassword)
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute('SELCET COUNT(*) FROM results')
    s = "<table style='border:1px solid red'>"

    for row in cursor:
        s = s + "<tr>"
    for x in row:
        s = s + "<td>" + str(x) + "</td>"
    s = s + "</tr>"
    conn.close()

    return "<html><body>" + s + "</body></html>"


@app.route('/commands')
def commands():
    producer = KafkaProducer(bootstrap_servers=cmdline_args.brokers,
                             value_serializer=lambda x:
                             dumps(x).encode('utf-8'))
    message_dict = {'command': 'Download', 'url': 'TODO'}
    producer.send(cmdline_args.topic, value=message_dict)

    return "<html><body>TODO: Send command</body></html>"
#    message_dict = {'command':'Download', 'url':url_string}
#    logging.info('Sending message {}'.format(str(i)))
#    producer.send(cmdline_args.topic, value=message_dict)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info('brokers={}'.format(cmdline_args.brokers))
    logging.info('topic={}'.format(cmdline_args.topic))
    logging.info('starting flask server')
    app.run(host='0.0.0.0', port=8080)


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
    main()
    logging.info('exiting')
