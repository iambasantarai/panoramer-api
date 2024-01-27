import os
import cv2
import time
import shutil
import imutils
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from stitcher import Stitcher
from flask_cors import CORS
from panoramer import *

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])


def allowed_file(filename):
    return "." in filename and filename.split(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def stitch_images(images):
    stitcher = Stitcher()

    no_of_images = len(images)

    # Modify the images width and height to keep the aspect ratio the same across images
    for i in range(no_of_images):
        images[i] = imutils.resize(images[i], width=800)

    for i in range(no_of_images):
        images[i] = imutils.resize(images[i], height=800)

    if no_of_images == 2:
        (result, matched_points) = stitcher.image_stitch(
            [images[0], images[1]], match_status=True
        )
    else:
        (result, matched_points) = stitcher.image_stitch(
            [images[no_of_images - 2], images[no_of_images - 1]], match_status=True
        )
        for i in range(no_of_images - 2):
            (result, matched_points) = stitcher.image_stitch(
                [images[no_of_images - i - 3], result], match_status=True
            )

    # Save stitcher and matched_points images in the 'static/output' folder
    output_folder = "static/output"
    os.makedirs(output_folder, exist_ok=True)

    panorama_path = os.path.join(output_folder, "panorama_image.jpg")
    matched_points_path = os.path.join(output_folder, "matched_points.jpg")

    cv2.imwrite(panorama_path, result)
    cv2.imwrite(matched_points_path, matched_points)

    return panorama_path, matched_points_path


@app.route("/heartbeat", methods=["GET"])
def heartbeat():
    heartbeat = time.monotonic_ns()
    return {"heartbeat": heartbeat}, 200


@app.route("/upload", methods=["POST"])
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
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                is_uploaded = True
            else:
                return jsonify({"message": "Invalid file type!"}), 400

        if len(files) < 2:
            return jsonify({"message": "At least two images are required!"}), 400

        if is_uploaded:
            return jsonify({"message": "Images uploaded successfully!"}), 201

    except Exception as e:
        return jsonify({"message": e}), 500


@app.route("/stitch-opencv", methods=["GET"])
def stitch_api():
    try:
        # Get the list of uploaded files from the 'uploads' directory
        uploaded_files = os.listdir(UPLOAD_FOLDER)

        if len(uploaded_files) < 2:
            return (
                jsonify({"message": "At least two images are required for stitching!"}),
                400,
            )

        filename = [
            os.path.join(UPLOAD_FOLDER, file_name) for file_name in uploaded_files
        ]
        images = [cv2.imread(file) for file in filename]

        panorama_path, matched_points_path = stitch_images(images)

        return (
            jsonify(
                {
                    "message": "Images stitched successfully!",
                    "panorama_image_path": panorama_path,
                    "matched_points_path": matched_points_path,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/generate-panorama", methods=["POST"])
def generate_panorama():
    try:
        uploaded_files = os.listdir(UPLOAD_FOLDER)

        # sorting files
        uploaded_files = sorted(uploaded_files, key=lambda x: int(Path(x).stem))

        if len(uploaded_files) < 2:
            return (
                jsonify({"message": "At least two images are required for stitching!"}),
                400,
            )

        print(uploaded_files)

        panorama = Panoramer(parent_folder=UPLOAD_FOLDER, img_name_list=uploaded_files)
        panorama.generate()

        # get list of generated files
        results_folder = os.path.join(UPLOAD_FOLDER, "results")
        generated_files = os.listdir(results_folder)

        # group the files based on their names
        grouped_results = {
            "sift_correspondences": [],
            "inliers_outliers": [],
            "panoramas": [],
        }

        for file_name in generated_files:
            if "sift_correspondence" in file_name:
                grouped_results["sift_correspondences"].append(
                    os.path.join(results_folder, file_name)
                )
            elif "inliers" in file_name or "outliers" in file_name:
                grouped_results["inliers_outliers"].append(
                    os.path.join(results_folder, file_name)
                )
            elif "panorama" in file_name:
                grouped_results["panoramas"].append(
                    os.path.join(results_folder, file_name)
                )

            # sort files withing each group
            for category, files in grouped_results.items():
                grouped_results[category] = sorted(files)

        return jsonify(
            {
                "message": "Panorama generated successfully!",
                "results": grouped_results,
            },
            200,
        )
    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/clear-uploads", methods=["DELETE"])
def delete_uploads():
    try:
        # Clear the contents of the 'uploads' folder
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        return jsonify({"message": "Uploads folder cleared successfully!"}), 200

    except Exception as e:
        return jsonify({"message": str(e)}), 500


@app.route("/serve-files/<path:filename>")
def serve_files(filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER, RESULTS_FOLDER), filename)


if __name__ == "__main__":
    app.run(debug=True)
