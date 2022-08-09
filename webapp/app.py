import os
import logging
import argparse
import json
from flask import Flask, request, render_template
import psycopg2
from psycopg2.extras import RealDictCursor
from kafka import KafkaProducer
from json import dumps


cmdline_args = []
app = Flask(__name__, static_folder='static', static_url_path='')


@app.route('/results', methods=["GET", "PUT"])
def results():
    if request.method == "PUT":
        data = request.get_json(force=True)
        with open('./temp.json', 'w+') as f:
            json.dump(data, f)
    elif request.method == "GET":
        with open('./temp.json', 'r') as f:
            data = json.load(f)
        # cluster_dict = json.loads(data)
        return render_template("main.html", clusters=list(data.keys()))

    # conn = psycopg2.connect(
    #     host=cmdline_args.dbhost,
    #     port=5432,
    #     dbname=cmdline_args.dbname,
    #     user=cmdline_args.dbusername,
    #     password=cmdline_args.dbpassword)
    # cursor = conn.cursor(cursor_factory=RealDictCursor)
    # cursor.execute('SELECT COUNT(*) FROM results')
    # res = cursor.fetchone()

    # s = "<table style='border:1px solid red'><tr><td>Count result: "
    # s = s + str(res['count']) + "</td></tr>"
    # conn.close()

    # return "<html><body>" + s + "</body></html>"


@app.route("/cluster/<images>")
def cluster(images):
    """
    This function gets triggered when you try to view a particular cluster.

    You can chose to associate the clusters with numbers (n), but right now w are not
    doing that. We just read in all the clusters, and chose the nth one to send back.

    Essentially you need to figure out where the clusters are, and get a list of filepaths
    to the images you want to show. Then, pass that list of filepaths for the images to
    render_template at the very end. Right now I am passing tuples into the list, but you 
    can change that if you want in the template file (cluster.html)
    """
    # n = find_new_cluster(n)
    filepaths = []

    # open the cluster data and get all the filepaths for images in the cluster here. 
    # will change based on how the clusters are organized. 
    # with open(json_results_file) as f:
    #     data = json.loads(f.read())
    for i, file in enumerate(images):
        print(file.replace(':', '/'))
        filepaths.append((i, file.replace(':', '/')))

    """
    Here, you can filter out the number of images you want to display
    if you are having issues with speed in the UI
    """
    #filepaths = filepaths[20] # first 200
    """
    By now, filepaths should be something like...
    [
        (1, "/home/images/cluster1/image1.jpg"),
        (2, "/home/images/cluster1/image2.jpg"),
        ...
    ]
    """
    return render_template("cluster.html", filepaths=filepaths, n=-1)


@app.route('/commands', methods=["GET", "POST"])
def commands():
    if request.method == "POST":
        producer = KafkaProducer(bootstrap_servers=cmdline_args.brokers,
                                 value_serializer=lambda x:
                                 dumps(x).encode('utf-8'))
        command = request.form.get("command")
        if command == 'Download':
            url = request.form.get("url")
            scan = request.form.get('scannsfw')
            app.logger.info(url)
            app.logger.info(scan)
            message_dict = {'command': command, 'url': url, 'scan': scan}
            producer.send(cmdline_args.topic, value=message_dict)
        elif command == 'Cluster':
            start_date = request.form.get("startd")
            end_date = request.form.get("endd")
            app.logger.info(start_date)
            app.logger.info(end_date)
            message_dict = {'command': command, 'startDate': start_date, 'endDate': end_date}
            producer.send(cmdline_args.topic, value=message_dict)


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
