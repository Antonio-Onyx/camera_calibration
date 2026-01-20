import os
import glob
import numpy as np
import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
from sklearn.cluster import KMeans
import threading

# Configuration constants from notebook
ROWS = 6
COLS = 9
SPACING = 11.5  # mm
HORIZONTAL_DISTANCE = 23.5
VERTICAL_DISTANCE = 26.5
SQUARE_SIZE = HORIZONTAL_DISTANCE / 2

# Global Wine Theme Colors
COLOR_DARK_BG = "#1a1a1a"
COLOR_WINE_ACCENT = "#800020"
COLOR_WINE_HOVER = "#4a0404"
COLOR_TEXT_MAIN = "#ffffff"
COLOR_TEXT_SECONDARY = "#b0b0b0"


class CalibrationApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CanSat Camera Calibration")
        self.geometry("900x700")
        self.configure(fg_color=COLOR_DARK_BG)

        # Initialize variables
        self.objp = self._generate_object_points()
        self.valid_objpoints = []
        self.valid_imgpoints = []
        self.image_paths = []
        self.results = {}

        self._setup_ui()

    def _generate_object_points(self):
        objp = np.zeros((ROWS * COLS, 3), np.float32)
        for i in range(ROWS):
            for j in range(COLS):
                objp[i * COLS + j] = [
                    ((2 * j) + (i % 2)) * SQUARE_SIZE,
                    i * VERTICAL_DISTANCE,
                    0,
                ]
        return objp

    def _setup_ui(self):
        # Header
        self.header_frame = ctk.CTkFrame(
            self, fg_color=COLOR_WINE_ACCENT, height=80, corner_radius=0
        )
        self.header_frame.pack(fill="x", side="top")

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="CAMERA CALIBRATION TOOL",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
        )
        self.title_label.place(relx=0.5, rely=0.5, anchor="center")

        # Main Container
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=40, pady=20)

        # Left Panel - Actions
        self.actions_frame = ctk.CTkFrame(
            self.main_frame, fg_color="#262626", width=250
        )
        self.actions_frame.pack(side="left", fill="y", padx=(0, 20), pady=0)

        self.select_btn = ctk.CTkButton(
            self.actions_frame,
            text="Select Image Folder",
            command=self._select_folder,
            fg_color=COLOR_WINE_ACCENT,
            hover_color=COLOR_WINE_HOVER,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.select_btn.pack(pady=(40, 20), padx=20)

        self.run_btn = ctk.CTkButton(
            self.actions_frame,
            text="Run Calibration",
            command=self._start_calibration_thread,
            fg_color=COLOR_WINE_ACCENT,
            hover_color=COLOR_WINE_HOVER,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
        )
        self.run_btn.pack(pady=20, padx=20)

        self.folder_label = ctk.CTkLabel(
            self.actions_frame,
            text="No folder selected",
            text_color=COLOR_TEXT_SECONDARY,
            wraplength=200,
        )
        self.folder_label.pack(pady=20, padx=20)

        self.progress_bar = ctk.CTkProgressBar(
            self.actions_frame, fg_color="#404040", progress_color=COLOR_WINE_ACCENT
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=20, padx=20, fill="x")

        # Right Panel - Results
        self.results_frame = ctk.CTkFrame(self.main_frame, fg_color="#262626")
        self.results_frame.pack(side="right", fill="both", expand=True)

        self.results_label = ctk.CTkLabel(
            self.results_frame,
            text="CALIBRATION RESULTS",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLOR_WINE_ACCENT,
        )
        self.results_label.pack(pady=(20, 10))

        self.stats_text = ctk.CTkTextbox(
            self.results_frame,
            fg_color="#1a1a1a",
            text_color=COLOR_TEXT_MAIN,
            font=ctk.CTkFont(family="Consolas", size=14),
            border_width=1,
            border_color="#404040",
        )
        self.stats_text.pack(fill="both", expand=True, padx=20, pady=20)
        self.stats_text.insert("0.0", "Waiting for calibration...")

    def _select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.image_paths = glob.glob(os.path.join(folder, "*.jpg"))
            self.folder_label.configure(
                text=f"Selected: {os.path.basename(folder)}\n({len(self.image_paths)} images)"
            )
            if self.image_paths:
                self.run_btn.configure(state="normal")
            else:
                messagebox.showwarning(
                    "Warning", "No .jpg images found in this folder."
                )

    def _start_calibration_thread(self):
        self.run_btn.configure(state="disabled")
        self.select_btn.configure(state="disabled")
        self.stats_text.delete("0.0", "end")
        self.stats_text.insert("0.0", "Processing images... Please wait.\n")
        threading.Thread(target=self._run_calibration, daemon=True).start()

    def _run_calibration(self):
        try:
            valid_objpts, valid_imgpts, ratios, dy_stds, failed = (
                self._evaluate_images_logic(self.image_paths)
            )

            total_imgs = len(self.image_paths)
            valid_imgs = len(valid_imgpts)

            results_str = f"=== SUMMARY ===\n"
            results_str += f"Images Valid: {valid_imgs} of {total_imgs}\n"

            if ratios:
                results_str += f"Ratio Mean: {np.mean(ratios):.4f}\n"
                results_str += f"Ratio Std:  {np.std(ratios):.4f}\n"
                results_str += f"DY Mean:    {np.mean(dy_stds):.4f}\n"
                results_str += f"DY Std:     {np.std(dy_stds):.4f}\n"
                results_str += "----------------------------\n"

            if valid_imgs < 10:
                results_str += (
                    "\nWARNING: Not enough valid images for reliable calibration.\n"
                )
            else:
                # Perform Camera Calibration
                img = cv2.imread(self.image_paths[0])
                gray_size = img.shape[:2][::-1]

                ret, k, dist, rvecs, tvecs = cv2.calibrateCamera(
                    valid_objpts, valid_imgpts, gray_size, None, None
                )

                results_str += f"RMS Error: {ret:.2f}\n\n"
                results_str += "INTRINSIC MATRIX (K):\n"
                results_str += np.array2string(k, precision=2, suppress_small=True)
                results_str += "\n\nDISTORTION COEFFICIENTS:\n"
                results_str += np.array2string(dist, precision=4)

                # Save results automatically
                np.savez(
                    "calibration_results.npz",
                    K=k,
                    dist=dist,
                    rvecs=rvecs,
                    tvecs=tvecs,
                    rms=ret,
                )
                results_str += (
                    "\n\n[SUCCESS] Calibration saved to 'calibration_results.npz'"
                )

            self._update_ui_results(results_str)

        except Exception as e:
            self._update_ui_results(f"ERROR: {str(e)}")
        finally:
            self.run_btn.configure(state="normal")
            self.select_btn.configure(state="normal")

    def _update_ui_results(self, text):
        self.stats_text.delete("0.0", "end")
        self.stats_text.insert("0.0", text)
        self.progress_bar.set(0)

    def _evaluate_images_logic(self, img_list):
        ratios, dy_stds, failed = [], [], []
        valid_objpoints = []
        valid_imgpoints = []

        # Detector configuration
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.filterByColor = True
        blob_params.blobColor = 0
        blob_params.filterByArea = True
        blob_params.minArea = 25
        blob_params.maxArea = 800
        blob_params.filterByCircularity = False
        blob_params.filterByConvexity = False
        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = 0.01
        blob_params.minThreshold = 10
        blob_params.maxThreshold = 200
        blob_params.thresholdStep = 10
        detector = cv2.SimpleBlobDetector_create(blob_params)

        total = len(img_list)
        for i, img_path in enumerate(img_list):
            self.progress_bar.set((i + 1) / total)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints = detector.detect(gray)

            if len(keypoints) != ROWS * COLS:
                failed.append(img_path)
                continue

            pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            ys = pts[:, 1].reshape(-1, 1)

            kmeans = KMeans(n_clusters=ROWS, n_init=10, random_state=0)
            row_labels = kmeans.fit_predict(ys)
            row_centers = kmeans.cluster_centers_.flatten()
            row_order = np.argsort(row_centers)

            ordered_points_list = []
            for row_idx in row_order:
                row_pts = pts[row_labels == row_idx]
                row_pts = row_pts[np.argsort(row_pts[:, 0])]
                ordered_points_list.append(row_pts)

            # Validation 3
            dx_even = np.diff(ordered_points_list[0][:, 0]).mean()
            offset = ordered_points_list[1][0, 0] - ordered_points_list[0][0, 0]
            ratio = offset / dx_even

            if not (0.3 < ratio < 0.7):
                failed.append(img_path)
                continue

            # Validation 4
            row_centers_y = [row[:, 1].mean() for row in ordered_points_list]
            dy = np.diff(row_centers_y)
            dy_metric = np.std(dy) / np.mean(dy)

            if dy_metric > 0.1:
                failed.append(img_path)
                continue

            ratios.append(ratio)
            dy_stds.append(dy_metric)

            ordered_points_final = np.vstack(ordered_points_list).astype(np.float32)
            valid_objpoints.append(self.objp.copy())
            valid_imgpoints.append(ordered_points_final.reshape(-1, 1, 2))

        return valid_objpoints, valid_imgpoints, ratios, dy_stds, failed


if __name__ == "__main__":
    app = CalibrationApp()
    app.mainloop()
