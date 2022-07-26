from flask import Flask, render_template, send_file, url_for, request
import sys
import os
import json

# TODO read in a bunch of cluster files, turn them into lists

app = Flask(__name__)

@app.route("/")
def main():
    with open('./ukr_clusters.json') as f:
        data = json.loads(f.read())
    return render_template("main.html", clusters=list(data.keys()))

@app.route("/cluster/<n>")
def cluster(n):
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
    with open('./ukr_clusters.json') as f:
        data = json.loads(f.read())
        for i, file in enumerate(data[str(n)]):
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
    return render_template("cluster.html", filepaths=filepaths, n=n)

@app.route("/<path:filepath>")
def get_image(filepath):
    """
    This function sends the image to the front end
    You need to ensure that the filepath that gets passed in is correct,
    or parse it accordingly.

    Make sure that the folder with the images is mounted and readable
    """
    # filepath = filepath.replace("world/", "")
    # filepath = "/media/wtheisen/scratch2/clusterImages/" + filepath
    # filepath = "/media/jbrogan4/" + filepath
    return send_file(filepath)

# TBH IDK what this is doing lmao
@app.route("/results", methods=["POST"])
def results():
    try:
        parse_result(request.json)
        return json.dumps({"status": "success"})
    except Exception as e:
        print(e)
        return json.dumps({"status": "failure"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
