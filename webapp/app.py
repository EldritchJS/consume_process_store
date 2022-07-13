import os
import logging
import argparse
from flask import Flask, request, render_template
import psycopg2
from psycopg2.extras import RealDictCursor
from kafka import KafkaProducer
from json import dumps


cmdline_args = []
app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/results')
def results():
    conn = psycopg2.connect(
        host=cmdline_args.dbhost,
        port=5432,
        dbname=cmdline_args.dbname,
        user=cmdline_args.dbusername,
        password=cmdline_args.dbpassword)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute('SELECT COUNT(*) FROM results')
    res = cursor.fetchone()

    s = "<table style='border:1px solid red'><tr><td>Count result: "
    s = s + str(res['count']) + "</td></tr>"
    conn.close()

    return "<html><body>" + s + "</body></html>"


@app.route('/commands', methods=["GET", "POST"])
def commands():
    if request.method == "POST":
#        producer = KafkaProducer(bootstrap_servers=cmdline_args.brokers,
 #                                value_serializer=lambda x:
  #                               dumps(x).encode('utf-8'))
        command = request.form.get("command")
        if command == 'Download':
            url = request.form.get("url")
            scan = request.form.get('scannsfw')
            app.logger.info(url)
            app.logger.info(scan)
            message_dict = {'command': command, 'url': url, 'scan': scan}
#            producer.send(cmdline_args.topic, value=message_dict)
        elif command == 'Cluster':
            start_date = request.form.get("startd")
            end_date = request.form.get("endd")
            app.logger.info(start_date)
            app.logger.info(end_date)
            message_dict = {'command': command, 'startDate': start_date, 'endDate': end_date}
#            producer.send(cmdline_args.topic, value=message_dict)
        

        return "<html><body>Sent Command: " + command + "</body></html>"
    return render_template("command_form.html")


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
