from ransac import *
from match_features import *
from scipy import optimize
from optimization import *
from crop import crop

class Panoramer:
    def __init__(self, parent_folder, img_name_list):
        self.img_all = {}
        self.parent_folder = parent_folder
        self.img_name_list = img_name_list
        self.middle_id = int(np.floor(len(img_name_list) / 2))

    def generate(self):
        H_all = {}
        for i in range(len(self.img_name_list) - 1):
            print(
                "#### Processing {} & {} ####".format(
                    self.img_name_list[i], self.img_name_list[i + 1]
                )
            )

            key = "H{}{}".format(i, i + 1)

            image_1_path = os.path.join(self.parent_folder, self.img_name_list[i])
            image_2_path = os.path.join(self.parent_folder, self.img_name_list[i + 1])

            # Get SIFT descriptors
            siftmatch_obj = SiftFeatureMatching(
                image_1_path, image_2_path, result_dir="", nfeatures=2000, gamma=0.6
            )
            correspondence = siftmatch_obj.run()

            # Run RANSAC to remove outliers
            ransac_obj = RANSAC()
            inliers_cnt, inliers, outliers, sample_pts, final_H = ransac_obj.run_ransac(
                correspondence
            )

            result_path = os.path.join(
                siftmatch_obj.result_dir, siftmatch_obj.prefix + "_inliers.jpg"
            )
            ransac_obj.draw_lines(
                np.concatenate((inliers, sample_pts), axis=0),
                siftmatch_obj.image_1_bgr,
                siftmatch_obj.image_2_bgr,
                result_path,
                line_color=RANSAC._GREEN,
                pt_color=[0, 0, 0],
            )

            result_path = os.path.join(
                siftmatch_obj.result_dir, siftmatch_obj.prefix + "_outliers.jpg"
            )
            ransac_obj.draw_lines(
                outliers,
                siftmatch_obj.image_1_bgr,
                siftmatch_obj.image_2_bgr,
                result_path,
                line_color=RANSAC._RED,
                pt_color=[0, 0, 0],
            )

            # Optimize the homography using Levenberg-Marquardt optimization
            x = np.concatenate((inliers, sample_pts), axis=0)
            opt_obj = OptimizationFunction(
                fun=fun_LM_homography,
                x0=final_H.flatten(),
                jac=jac_LM_homography,
                args=(x[:, 0:2], x[:, 2:]),
            )
            LM_sol = opt_obj.levenberg_marquardt(delta_thresh=1e-24, tau=0.8)

            H_all[key] = LM_sol.x.reshape(3, 3)
            H_all[key] = H_all[key] / H_all[key][-1, -1]

        H_all = self.compute_H_wrt_middle_img(H_all)

        self.stitch(H_all, siftmatch_obj.result_dir)

    def stitch(self, H_all, result_dir):
        canvas_img, mask, offset = self.get_blank_canvas(H_all)

        for i, img_name in enumerate(self.img_name_list):
            key = "H{}{}".format(i, self.middle_id)
            H = H_all[key]
            img_path = os.path.join(self.parent_folder, img_name)

            img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            canvas_img = fit_image_in_target_space(
                img_rgb, canvas_img, mask, np.linalg.inv(H), offset=offset
            )  # the inp to fit_image_in_target_space
            # pts_in_img_2 = H * pts_in_canvas
            mask[np.where(canvas_img)[0:2]] = 0

            result_path = os.path.join(result_dir, "panorama_{}.jpg".format(i))
            cv2.imwrite(result_path, canvas_img[:, :, (2, 1, 0)])

            if i == len(self.img_name_list) - 1:
                final_panorama_path = os.path.join(result_dir, "final_panorama.jpg")
                crop(result_path, final_panorama_path)
                print(f"Final panorama image saved at: {final_panorama_path}")

    def get_blank_canvas(self, H_all):
        img_path = os.path.join(self.parent_folder, self.img_name_list[0])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        img_h, img_w, _ = img.shape

        min_crd_canvas = np.array([np.inf, np.inf, np.inf])
        max_crd_canvas = np.array([-np.inf, -np.inf, -np.inf])

        for i in range(len(self.img_name_list)):
            key = "H{}{}".format(i, self.middle_id)
            H = H_all[key]
            min_crd, max_crd = self.compute_extent(H, img_w, img_h)

            min_crd_canvas = np.minimum(min_crd, min_crd_canvas)
            max_crd_canvas = np.maximum(max_crd, max_crd_canvas)

        width_canvas = np.ceil(max_crd_canvas - min_crd_canvas)[0] + 1
        height_canvas = np.ceil(max_crd_canvas - min_crd_canvas)[1] + 1

        canvas_img = np.zeros(
            (int(height_canvas), int(width_canvas), 3), dtype=np.int64
        )

        offset = min_crd_canvas.astype(np.int64)
        offset[2] = 0  # [x_offset, y_offset, 0]

        mask = np.ones((int(height_canvas), int(width_canvas)))

        return canvas_img, mask, offset

    def compute_extent(self, H, img_w, img_h):
        corners_img = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])

        t_one = np.ones((corners_img.shape[0], 1))
        t_out_pts = np.concatenate((corners_img, t_one), axis=1)
        canvas_crd_corners = np.matmul(H, t_out_pts.T)
        canvas_crd_corners = (
            canvas_crd_corners / canvas_crd_corners[-1, :]
        )  # cols of [x1, y1, z1]

        min_crd = np.amin(canvas_crd_corners.T, axis=0)  # [x, y, z]
        max_crd = np.amax(canvas_crd_corners.T, axis=0)
        return min_crd, max_crd

    def compute_H_wrt_middle_img(self, H_all):
        num_imgs = len(H_all) + 1

        key = "H{}{}".format(self.middle_id, self.middle_id)
        H_all[key] = np.eye(3)

        for i in range(0, self.middle_id):
            key = "H{}{}".format(i, self.middle_id)
            j = i
            temp = np.eye(3)
            while j < self.middle_id:
                key_t = "H{}{}".format(j, j + 1)
                temp = np.matmul(H_all[key_t], temp)
                j += 1

            H_all[key] = temp

        for i in range(self.middle_id + 1, num_imgs):
            key = "H{}{}".format(i, self.middle_id)

            temp = np.eye(3)

            j = i - 1

            while j >= self.middle_id:
                key_t = "H{}{}".format(j, j + 1)
                temp = np.matmul(np.linalg.inv(H_all[key_t]), temp)
                j -= 1

            H_all[key] = temp

        return H_all
