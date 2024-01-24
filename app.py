import os
import time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(["png", "jpg"])

def allowed_file(filename):
    return "." in filename and filename.split(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/heartbeat", methods = ["GET"])
def heartbeat():
    heartbeat = time.monotonic_ns()
    return {"heartbeat": heartbeat}, 200

@app.route("/upload", methods=['POST'])
def upload_image():
    try:
        # check if the post request contains files
        if "files[]" not in request.files:
            return jsonify({"message": "Images are required!"}), 400

        files = request.files.getlist("files[]")

        print("+++ uploaded files +++")
        print(len(files))
        for file in files:
            print(f"FILE: {file.filename}")
        print("+++ uploaded files +++")

        is_uploaded = False

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                is_uploaded = True
            else:
                return jsonify({"message": "Invalid file type!"}), 400

        if len(files) < 2:
            return jsonify({"message": "At least two images are required!"}), 400

        if is_uploaded:
            return jsonify({"message": "Images uploaded successfully!"}), 201

    except Exception as e:
        return jsonify({"message": e}), 500

if __name__ == "__main__":
    app.run(debug=True)
