#!/usr/bin/env python3
# camshift_annotator.py
# Single-file MVP: capture, ArUco/ChArUco pose, base/prev deltas, CSV log, 
# optional frame dumps.


import argparse, os, time, math, sys, csv, pathlib, json
from datetime import datetime
import numpy as np
import cv2 as cv


# ---------- Utilities ----------
def rodrigues_to_euler_zyx(rvec):
    R, _ = cv.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        yaw   = math.atan2(R[1,0], R[0,0])          # Z
        pitch = math.atan2(-R[2,0], sy)             # Y
        roll  = math.atan2(R[2,1], R[2,2])          # X
    else:
        yaw   = math.atan2(-R[0,1], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        roll  = 0.0
    return roll, pitch, yaw  # radians


def se3_from_rvec_tvec(rvec, tvec):
    R, _ = cv.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = tvec.flatten()
    return T


def se3_inv(T):
    R = T[:3,:3]; t = T[:3,3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3,:3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def se3_mul(A, B):
    return A @ B


def rpy_deg(rpy_rad):
    return tuple(np.degrees(np.array(rpy_rad)))


def write_yaml_camera(path, K, dist, size):
    fs = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K)
    fs.write("dist_coeffs", dist)
    fs.write("image_width", int(size[0]))
    fs.write("image_height", int(size[1]))
    fs.release()


def read_yaml_camera(path):
    fs = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None
    K  = fs.getNode("camera_matrix").mat()
    d  = fs.getNode("dist_coeffs").mat()
    w  = int(fs.getNode("image_width").real())
    h  = int(fs.getNode("image_height").real())
    fs.release()
    return K, d, (w,h)


def now_ns():
    return time.time_ns()


def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


# ---------- Boards ----------
def make_charuco_board(squares_x, squares_y, square_len_m, marker_len_m, dict_name):
    aruco = cv.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    board = aruco.CharucoBoard((squares_x, squares_y), square_len_m, marker_len_m, dictionary)
    # Create detector for OpenCV 4.7.0+ API
    detector = aruco.ArucoDetector(dictionary)
    return dictionary, board, detector


def render_board_image(board, px=2480, py=3508):  # A4 at ~300dpi default
    img = board.generateImage((px, py), marginSize=int(0.02*px), borderBits=1)
    return img


def save_board_png(board_img, path):
    cv.imwrite(path, board_img)


def make_gridboard(markers_x, markers_y, marker_len_m, sep_len_m, dict_name):
    aruco = cv.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    board = aruco.GridBoard((markers_x, markers_y), marker_len_m, sep_len_m, dictionary)
    # Create detector for OpenCV 4.7.0+ API
    detector = aruco.ArucoDetector(dictionary)
    return dictionary, board, detector


# ---------- Calibration (ChArUco) ----------
def run_charuco_calibration(cap, detector, charuco_board, out_yaml, max_frames=25, save_dir=None):
    """Run interactive calibration, optionally saving images to disk.

    Args:
        cap: VideoCapture object or None if loading from images
        detector: ArucoDetector instance
        charuco_board: CharucoBoard instance
        out_yaml: Path to save calibration results
        max_frames: Maximum frames to collect
        save_dir: Optional directory to save calibration images
    """
    print("[Calib] Press 'c' to collect, 'q' to finish.")
    all_corners = []
    all_ids = []
    all_images = []  # Store grayscale images for calibration
    imsize = None
    charuco_detector = cv.aruco.CharucoDetector(charuco_board)

    # Setup image saving if requested
    if save_dir:
        ensure_dir(save_dir)
        print(f"[Calib] Saving images to {save_dir}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Calib] Camera read failed.")
            break
        vis = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # New API: detector.detectMarkers()
        corners, ids, _ = detector.detectMarkers(gray)
        if len(corners) > 0:
            cv.aruco.drawDetectedMarkers(vis, corners, ids)
            # New API: charuco_detector.detectBoard()
            ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
            if ch_corners is not None and ch_ids is not None:
                cv.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
                cv.putText(vis, f"Charuco corners: {len(ch_ids)}", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv.imshow("Calibration", vis)
        k = cv.waitKey(1) & 0xFF
        if k == ord('c') and 'ch_corners' in locals() and ch_corners is not None and ch_ids is not None and len(ch_ids) >= 6:
            all_corners.append(ch_corners)
            all_ids.append(ch_ids)
            all_images.append(gray.copy())
            imsize = gray.shape[::-1]

            # Save image immediately to prevent data loss
            if save_dir:
                img_path = os.path.join(save_dir, f"calib_{len(all_corners):03d}.png")
                cv.imwrite(img_path, gray)
                print(f"[Calib] Saved {img_path}")

            print(f"[Calib] Collected {len(all_corners)}/{max_frames}")
            if len(all_corners) >= max_frames:
                print("[Calib] Reached max frames.")
                break
        elif k == ord('q'):
            break
    cv.destroyWindow("Calibration")
    if len(all_corners) < 5:
        print("[Calib] Not enough views for calibration.")
        return None

    # OpenCV 4.7.0+ API: Use standard calibrateCamera with ChArUco corners
    # Get board's 3D object points (corner positions in board coordinate frame)
    board_obj_points = charuco_board.getChessboardCorners()

    # Prepare data for calibrateCamera
    object_points = []  # 3D points in board frame
    image_points = []   # 2D points in image frame

    for corners, ids in zip(all_corners, all_ids):
        # Match detected corner IDs to board's object points
        obj_pts = []
        img_pts = []
        for i, corner_id in enumerate(ids.flatten()):
            obj_pts.append(board_obj_points[corner_id])
            img_pts.append(corners[i])
        object_points.append(np.array(obj_pts, dtype=np.float32))
        image_points.append(np.array(img_pts, dtype=np.float32))

    # Run standard camera calibration
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        object_points,      # 3D points in board frame
        image_points,       # 2D points in image frame
        imsize,             # Image size (width, height)
        None,               # Initial camera matrix (None = estimate)
        None)               # Initial distortion coeffs (None = estimate)

    if not ret:
        print("[Calib] Calibration failed.")
        return None

    write_yaml_camera(out_yaml, K, dist, imsize)
    print(f"[Calib] Saved to {out_yaml}")
    return K, dist, imsize


def run_charuco_calibration_from_images(image_dir, detector, charuco_board, out_yaml):
    """Run calibration using pre-captured images from a directory.

    Args:
        image_dir: Directory containing calibration images
        detector: ArucoDetector instance
        charuco_board: CharucoBoard instance
        out_yaml: Path to save calibration results

    Returns:
        Tuple of (K, dist, imsize) or None if calibration failed
    """
    import glob

    # Find all image files
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")) +
                        glob.glob(os.path.join(image_dir, "*.jpg")) +
                        glob.glob(os.path.join(image_dir, "*.jpeg")))

    if len(image_paths) == 0:
        print(f"[Error] No images found in {image_dir}")
        return None

    print(f"[Calib] Found {len(image_paths)} images in {image_dir}")

    all_corners = []
    all_ids = []
    imsize = None
    charuco_detector = cv.aruco.CharucoDetector(charuco_board)

    for i, img_path in enumerate(image_paths):
        # Read image
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[Warn] Failed to read {img_path}, skipping")
            continue

        imsize = img.shape[::-1]

        # Detect ChArUco board
        ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(img)

        if ch_corners is not None and ch_ids is not None and len(ch_ids) >= 6:
            all_corners.append(ch_corners)
            all_ids.append(ch_ids)
            print(f"[Calib] Processed {os.path.basename(img_path)}: {len(ch_ids)} corners detected")
        else:
            print(f"[Warn] {os.path.basename(img_path)}: insufficient corners detected, skipping")

    if len(all_corners) < 5:
        print(f"[Error] Not enough valid views for calibration (got {len(all_corners)}, need at least 5)")
        return None

    print(f"[Calib] Using {len(all_corners)} views for calibration")

    # Prepare data for calibrateCamera
    board_obj_points = charuco_board.getChessboardCorners()
    object_points = []
    image_points = []

    for corners, ids in zip(all_corners, all_ids):
        obj_pts = []
        img_pts = []
        for i, corner_id in enumerate(ids.flatten()):
            obj_pts.append(board_obj_points[corner_id])
            img_pts.append(corners[i])
        object_points.append(np.array(obj_pts, dtype=np.float32))
        image_points.append(np.array(img_pts, dtype=np.float32))

    # Run calibration
    ret, K, dist, rvecs, tvecs = cv.calibrateCamera(
        object_points,
        image_points,
        imsize,
        None,
        None)

    if not ret:
        print("[Calib] Calibration failed.")
        return None

    write_yaml_camera(out_yaml, K, dist, imsize)
    print(f"[Calib] Saved to {out_yaml}")
    return K, dist, imsize


# ---------- Pose estimation ----------
def estimate_pose_charuco(frame_gray, detector, charuco_board, K, dist):
    # Create CharucoDetector for board detection
    charuco_detector = cv.aruco.CharucoDetector(charuco_board)

    # Detect board and estimate pose
    ch_corners, ch_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame_gray)

    if ch_corners is None or ch_ids is None or len(ch_ids) < 6:
        return None

    # Get 3D object points for detected ChArUco corners
    board_obj_points = charuco_board.getChessboardCorners()
    obj_pts = []
    img_pts = []
    for i, corner_id in enumerate(ch_ids.flatten()):
        obj_pts.append(board_obj_points[corner_id])
        img_pts.append(ch_corners[i])

    obj_pts = np.array(obj_pts, dtype=np.float32)
    img_pts = np.array(img_pts, dtype=np.float32)

    # Use solvePnP for pose estimation
    ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, dist, flags=cv.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return rvec, tvec, len(ch_ids)


def estimate_pose_gridboard(frame_gray, detector, grid_board, K, dist):
    # New API: detector.detectMarkers()
    corners, ids, _ = detector.detectMarkers(frame_gray)
    if len(corners) == 0 or ids is None:
        return None

    # Get 3D object points for detected markers
    obj_points = grid_board.getObjPoints()

    # Match detected marker IDs to board's object points
    obj_pts = []
    img_pts = []
    for i, marker_id in enumerate(ids.flatten()):
        # Each marker has 4 corners
        if marker_id < len(obj_points) // 4:
            marker_obj_pts = obj_points[marker_id * 4:(marker_id + 1) * 4]
            for j in range(4):
                obj_pts.append(marker_obj_pts[j])
                img_pts.append(corners[i][0][j])

    if len(obj_pts) < 4:
        return None

    obj_pts = np.array(obj_pts, dtype=np.float32)
    img_pts = np.array(img_pts, dtype=np.float32)

    # Use solvePnP for pose estimation
    ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, dist, flags=cv.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return rvec, tvec, len(ids)


# ---------- Main loop ----------
def main():
    p = argparse.ArgumentParser(description="Cam-shift auto-annotator using ArUco/ChArUco.")
    p.add_argument("--source", type=str, default="0", help="Camera index or video path/URL. Default 0")
    p.add_argument("--mode", choices=["charuco","gridboard"], default="charuco", help="Detection mode")
    p.add_argument("--dict", type=str, default="DICT_4X4_50", help="cv.aruco predefined dictionary name")
    # ChArUco params
    p.add_argument("--charuco-squares-x", type=int, default=7)
    p.add_argument("--charuco-squares-y", type=int, default=5)
    p.add_argument("--square-m", type=float, default=0.035, help="Square size in meters")
    p.add_argument("--marker-m", type=float, default=0.026, help="Marker size in meters (ChArUco) or marker side (GridBoard)")
    # GridBoard params
    p.add_argument("--grid-markers-x", type=int, default=5)
    p.add_argument("--grid-markers-y", type=int, default=7)
    p.add_argument("--grid-sep-m", type=float, default=0.006, help="Separation between markers in meters")
    # Run options
    p.add_argument("--calib", type=str, default="camera.yaml", help="Path to camera YAML. If missing and mode=charuco, will calibrate.")
    p.add_argument("--out", type=str, default="run_out", help="Output directory")
    p.add_argument("--max-rate", type=float, default=20.0, help="Max capture Hz (<=20)")
    p.add_argument("--save-frames", action="store_true", help="Save annotated frames")
    p.add_argument("--frame-ext", type=str, default="jpg")
    p.add_argument("--draw", action="store_true", help="Show live window")
    p.add_argument("--board-png", action="store_true", help="Export board PNG to out/")
    # Calibration options
    p.add_argument("--save-calib-images", action="store_true", help="Save calibration images as they are captured")
    p.add_argument("--calib-images-dir", type=str, help="Directory to save/load calibration images (default: out/calib_images)")
    p.add_argument("--load-calib-images", action="store_true", help="Load calibration images from directory instead of camera")
    args = p.parse_args()

    if args.max_rate > 20.0:
        print("[Warn] Capped to 20 Hz.")
        args.max_rate = 20.0

    ensure_dir(args.out)
    frames_dir = os.path.join(args.out, "frames")
    if args.save_frames:
        ensure_dir(frames_dir)

    # Setup calibration images directory
    calib_images_dir = args.calib_images_dir if args.calib_images_dir else os.path.join(args.out, "calib_images")
    if args.save_calib_images or args.load_calib_images:
        ensure_dir(calib_images_dir)

    # Board setup
    if args.mode == "charuco":
        dictionary, board, detector = make_charuco_board(args.charuco_squares_x, args.charuco_squares_y, args.square_m, args.marker_m, args.dict)
        if args.board_png:
            img = render_board_image(board)
            save_board_png(img, os.path.join(args.out, "charuco_board.png"))
            print(f"[Info] Saved ChArUco board to {os.path.join(args.out,'charuco_board.png')}")
    else:
        dictionary, board, detector = make_gridboard(args.grid_markers_x, args.grid_markers_y, args.marker_m, args.grid_sep_m, args.dict)
        if args.board_png:
            canvas = 2000
            board_img = board.generateImage((canvas, canvas))
            save_board_png(board_img, os.path.join(args.out, "grid_board.png"))
            print(f"[Info] Saved GridBoard to {os.path.join(args.out,'grid_board.png')}")

    # Video source (only needed if not loading calibration from images)
    cap = None
    if not args.load_calib_images:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv.VideoCapture(src)
        if not cap.isOpened():
            print("[Error] Cannot open source.")
            sys.exit(1)

    # Calibration load or run
    calib_loaded = read_yaml_camera(args.calib)

    if calib_loaded is None and args.mode == "charuco":
        if args.load_calib_images:
            # Load calibration from pre-captured images
            print(f"[Info] Loading calibration images from {calib_images_dir}")
            calib_loaded = run_charuco_calibration_from_images(calib_images_dir, detector, board, args.calib)
            if calib_loaded is None:
                print("[Error] Calibration from images failed.")
                sys.exit(2)
        else:
            # Interactive calibration from camera
            print("[Info] Calibration file not found. Starting ChArUco calibration.")
            save_dir = calib_images_dir if args.save_calib_images else None
            calib_loaded = run_charuco_calibration(cap, detector, board, args.calib, max_frames=25, save_dir=save_dir)
            if calib_loaded is None:
                print("[Error] Calibration failed or aborted.")
                sys.exit(2)
            # Re-init stream after window usage
            cap.release()
            cap = cv.VideoCapture(src)
            if not cap.isOpened():
                print("[Error] Cannot reopen source after calibration.")
                sys.exit(3)
    elif calib_loaded is None and args.mode == "gridboard":
        print("[Error] GridBoard mode requires a camera YAML. Provide --calib.")
        sys.exit(4)

    # Ensure we have a camera open for the main loop
    if cap is None:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv.VideoCapture(src)
        if not cap.isOpened():
            print("[Error] Cannot open source for pose tracking.")
            sys.exit(5)

    K, dist, _ = calib_loaded

    # CSV
    csv_path = os.path.join(args.out, "poses.csv")
    with open(csv_path, "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow([
            "frame_idx","timestamp_ns","detected",
            "curr_tx_m","curr_ty_m","curr_tz_m","curr_roll_deg","curr_pitch_deg","curr_yaw_deg",
            "base_tx_m","base_ty_m","base_tz_m","base_roll_deg","base_pitch_deg","base_yaw_deg",
            "prev_tx_m","prev_ty_m","prev_tz_m","prev_roll_deg","prev_pitch_deg","prev_yaw_deg",
            "inliers_count"
        ])

        base_T = None
        prev_T = None
        target_dt = 1.0 / args.max_rate if args.max_rate > 0 else 0
        next_time = time.time()
        frame_idx = 0

        while True:
            # rate control
            now = time.time()
            if now < next_time:
                time.sleep(next_time - now)
            next_time += target_dt

            ok, frame = cap.read()
            if not ok:
                print("[Info] End of stream or read failure.")
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Pose
            est = (estimate_pose_charuco(gray, detector, board, K, dist) if args.mode=="charuco"
                   else estimate_pose_gridboard(gray, detector, board, K, dist))
            detected = est is not None
            if detected:
                rvec, tvec, n_inliers = est
                T = se3_from_rvec_tvec(rvec, tvec)
                roll, pitch, yaw = rpy_deg(rodrigues_to_euler_zyx(rvec))
                if base_T is None:
                    base_T = T.copy()
                curr_in_base = se3_mul(se3_inv(base_T), T)
                curr_in_prev = None
                if prev_T is not None:
                    curr_in_prev = se3_mul(se3_inv(prev_T), T)
                prev_T = T.copy()

                # deltas
                b_tx, b_ty, b_tz = curr_in_base[:3,3]
                b_r, b_p, b_y = rpy_deg(cv.Rodrigues(cv.Rodrigues(cv.Rodrigues(np.array([0,0,0],dtype=np.float64))[0])[0])[0])  # dummy to keep API parity
                # extract Euler from base delta:
                Rb = curr_in_base[:3,:3]
                # Convert to rvec then euler
                rbvec, _ = cv.Rodrigues(Rb)
                b_r, b_p, b_y = rpy_deg(rodrigues_to_euler_zyx(rbvec))

                if curr_in_prev is not None:
                    p_tx, p_ty, p_tz = curr_in_prev[:3,3]
                    Rp = curr_in_prev[:3,:3]
                    rpvec, _ = cv.Rodrigues(Rp)
                    p_r, p_p, p_y = rpy_deg(rodrigues_to_euler_zyx(rpvec))
                else:
                    p_tx=p_ty=p_tz=p_r=p_p=p_y = float("nan")

                # draw
                if args.draw or args.save_frames:
                    cv.drawFrameAxes(frame, K, dist, rvec, tvec, 0.05, 3)
                    cv.putText(frame, f"yaw(deg)={yaw:.2f}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                rvec=tvec=None
                roll=pitch=yaw = float("nan")
                b_tx=b_ty=b_tz=b_r=b_p=b_y = float("nan")
                p_tx=p_ty=p_tz=p_r=p_p=p_y = float("nan")
                n_inliers = 0

            ts = now_ns()
            wr.writerow([
                frame_idx, ts, int(detected),
                (0 if not detected else float(tvec[0])), (0 if not detected else float(tvec[1])), (0 if not detected else float(tvec[2])),
                roll, pitch, yaw,
                b_tx, b_ty, b_tz, b_r, b_p, b_y,
                p_tx, p_ty, p_tz, p_r, p_p, p_y,
                n_inliers
            ])

            if args.draw:
                cv.imshow("camshift_annotator", frame)
                key = cv.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
            if args.save_frames:
                outp = os.path.join(frames_dir, f"frame_{frame_idx:06d}.{args.frame_ext}")
                cv.imwrite(outp, frame)

            frame_idx += 1

    cap.release()
    if args.draw:
        cv.destroyAllWindows()
    print(f"[Done] CSV: {csv_path}")
    if args.save_frames:
        print(f"[Done] Frames dir: {frames_dir}")


if __name__ == "__main__":
    main()
