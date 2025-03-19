import os
import cv2
import numpy as np


class SiftKeyPointDescriptor:
    def __init__(self, kp, desc):
        self.kp = kp
        self.desc = desc


class SiftFeatureMatching:
    _BLUE = [255, 0, 0]
    _GREEN = [0, 255, 0]
    _RED = [0, 0, 255]
    _CYAN = [255, 255, 0]

    _line_thickness = 2
    _radius = 5
    _circ_thickness = 2

    def __init__(
        self, image_path_1, image_path_2, result_dir="", nfeatures=2000, gamma=0.8
    ):
        fname_1 = os.path.basename(image_path_1)
        fname_2 = os.path.basename(image_path_2)

        if not result_dir:
            result_dir = os.path.split(image_path_1)[0]

        self.result_dir = os.path.join(result_dir, "results")

        self.prefix = fname_1.split(".")[0] + "_" + fname_2.split(".")[0]

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        self.image_1_bgr = self.read_image(image_path_1)
        self.image_2_bgr = self.read_image(image_path_2)

        self.nfeatures = nfeatures
        self.gamma = gamma

    def read_image(self, image_path):
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return img_bgr

    def get_sift_features(self, img_bgr, nfeatures=2000):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Use the SIFT algorithm without xfeatures2d module
        sift_obj = cv2.SIFT_create(nfeatures)

        # kp_list_obj is a list of "KeyPoint" objects with location stored as tuple in "pt" attribute
        kp_list_obj, desc = sift_obj.detectAndCompute(image=img_gray, mask=None)

        kp = [x.pt for x in kp_list_obj]

        return SiftKeyPointDescriptor(kp, desc)

    def match_features(self, sift_kp_desc_obj1, sift_kp_desc_obj2, gamma=0.8):
        correspondence = []  # list of lists of [x1, y1, x2, y2]

        for i in range(len(sift_kp_desc_obj1.kp)):
            sc = np.linalg.norm(
                sift_kp_desc_obj1.desc[i] - sift_kp_desc_obj2.desc, axis=1
            )
            idx = np.argsort(sc)

            val = sc[idx[0]] / sc[idx[1]]

            if val <= gamma:
                correspondence.append(
                    [*sift_kp_desc_obj1.kp[i], *sift_kp_desc_obj2.kp[idx[0]]]
                )

        return correspondence

    def draw_correspondence(self, correspondence, imgage_1, image_2):
        if len(imgage_1.shape) == 2:
            imgage_1 = np.repeat(imgage_1[:, :, np.newaxis], 3, axis=2)

        if len(image_2.shape) == 2:
            image_2 = np.repeat(image_2[:, :, np.newaxis], 3, axis=2)

        h, w, _ = imgage_1.shape

        img_stack = np.hstack((imgage_1, image_2))

        for x1, y1, x2, y2 in correspondence:
            x1_d = int(round(x1))
            y1_d = int(round(y1))

            x2_d = int(round(x2) + w)
            y2_d = int(round(y2))

            cv2.circle(
                img_stack,
                (x1_d, y1_d),
                radius=self._radius,
                color=self._BLUE,
                thickness=self._circ_thickness,
                lineType=cv2.LINE_AA,
            )

            cv2.circle(
                img_stack,
                (x2_d, y2_d),
                radius=self._radius,
                color=self._BLUE,
                thickness=self._circ_thickness,
                lineType=cv2.LINE_AA,
            )

            cv2.line(
                img_stack,
                (x1_d, y1_d),
                (x2_d, y2_d),
                color=self._CYAN,
                thickness=self._line_thickness,
            )

        fname = os.path.join(self.result_dir, self.prefix + "_sift_correspondence.jpg")
        cv2.imwrite(fname, img_stack)

    def run(self):
        sift_kp_desc_obj1 = self.get_sift_features(
            self.image_1_bgr, nfeatures=self.nfeatures
        )
        sift_kp_desc_obj2 = self.get_sift_features(
            self.image_2_bgr, nfeatures=self.nfeatures
        )

        correspondence = self.match_features(
            sift_kp_desc_obj1, sift_kp_desc_obj2, gamma=self.gamma
        )

        self.draw_correspondence(correspondence, self.image_1_bgr, self.image_2_bgr)

        return correspondence
