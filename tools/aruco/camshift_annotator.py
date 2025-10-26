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
    return dictionary, board


def render_board_image(board, px=2480, py=3508):  # A4 at ~300dpi default
    img = board.generateImage((px, py), marginSize=int(0.02*px), borderBits=1)
    return img


def save_board_png(board_img, path):
    cv.imwrite(path, board_img)


def make_gridboard(markers_x, markers_y, marker_len_m, sep_len_m, dict_name):
    aruco = cv.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    board = aruco.GridBoard((markers_x, markers_y), marker_len_m, sep_len_m, dictionary)
    return dictionary, board


# ---------- Calibration (ChArUco) ----------
def run_charuco_calibration(cap, dictionary, charuco_board, out_yaml, max_frames=25):
    print("[Calib] Press 'c' to collect, 'q' to finish.")
    all_corners = []
    all_ids = []
    imsize = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Calib] Camera read failed.")
            break
        vis = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = cv.aruco.detectMarkers(gray, dictionary)
        if len(corners) > 0:
            cv.aruco.drawDetectedMarkers(vis, corners, ids)
            retval, ch_corners, ch_ids = cv.aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)
            if retval and ch_corners is not None and ch_ids is not None:
                cv.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
                cv.putText(vis, f"Charuco corners: {len(ch_ids)}", (20,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv.imshow("Calibration", vis)
        k = cv.waitKey(1) & 0xFF
        if k == ord('c') and 'ch_corners' in locals() and ch_corners is not None and ch_ids is not None and len(ch_ids) >= 6:
            all_corners.append(ch_corners)
            all_ids.append(ch_ids)
            imsize = gray.shape[::-1]
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
    ret, K, dist, rvecs, tvecs = cv.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=charuco_board,
        imageSize=imsize,
        cameraMatrix=None,
        distCoeffs=None)
    if not ret:
        print("[Calib] Calibration failed.")
        return None
    write_yaml_camera(out_yaml, K, dist, imsize)
    print(f"[Calib] Saved to {out_yaml}")
    return K, dist, imsize


# ---------- Pose estimation ----------
def estimate_pose_charuco(frame_gray, dictionary, charuco_board, K, dist):
    corners, ids, _ = cv.aruco.detectMarkers(frame_gray, dictionary)
    if len(corners) == 0 or ids is None:
        return None
    cv.aruco.refineDetectedMarkers(frame_gray, charuco_board, corners, ids, rejectedCorners=None, cameraMatrix=K, distCoeffs=dist)
    retval, ch_corners, ch_ids = cv.aruco.interpolateCornersCharuco(corners, ids, frame_gray, charuco_board, cameraMatrix=K, distCoeffs=dist)
    if not retval or ch_corners is None or ch_ids is None or len(ch_ids) < 6:
        return None
    ok, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, charuco_board, K, dist, None, None)
    if not ok:
        return None
    return rvec, tvec, len(ch_ids)


def estimate_pose_gridboard(frame_gray, dictionary, grid_board, K, dist):
    corners, ids, _ = cv.aruco.detectMarkers(frame_gray, dictionary)
    if len(corners) == 0 or ids is None:
        return None
    cv.aruco.refineDetectedMarkers(frame_gray, grid_board, corners, ids, rejectedCorners=None, cameraMatrix=K, distCoeffs=dist)
    ok, rvec, tvec = cv.aruco.estimatePoseBoard(corners, ids, grid_board, K, dist, None, None)
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
    args = p.parse_args()

    if args.max_rate > 20.0:
        print("[Warn] Capped to 20 Hz.")
        args.max_rate = 20.0

    ensure_dir(args.out)
    frames_dir = os.path.join(args.out, "frames")
    if args.save_frames:
        ensure_dir(frames_dir)

    # Board setup
    if args.mode == "charuco":
        dictionary, board = make_charuco_board(args.charuco_squares_x, args.charuco_squares_y, args.square_m, args.marker_m, args.dict)
        if args.board_png:
            img = render_board_image(board)
            save_board_png(img, os.path.join(args.out, "charuco_board.png"))
            print(f"[Info] Saved ChArUco board to {os.path.join(args.out,'charuco_board.png')}")
    else:
        dictionary, board = make_gridboard(args.grid_markers_x, args.grid_markers_y, args.marker_m, args.grid_sep_m, args.dict)
        if args.board_png:
            canvas = 2000
            board_img = board.generateImage((canvas, canvas))
            save_board_png(board_img, os.path.join(args.out, "grid_board.png"))
            print(f"[Info] Saved GridBoard to {os.path.join(args.out,'grid_board.png')}")

    # Video source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv.VideoCapture(src)
    if not cap.isOpened():
        print("[Error] Cannot open source.")
        sys.exit(1)

    # Calibration load or run
    calib_loaded = read_yaml_camera(args.calib)
    if calib_loaded is None and args.mode == "charuco":
        print("[Info] Calibration file not found. Starting ChArUco calibration.")
        calib_loaded = run_charuco_calibration(cap, dictionary, board, args.calib, max_frames=25)
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
            est = (estimate_pose_charuco(gray, dictionary, board, K, dist) if args.mode=="charuco"
                   else estimate_pose_gridboard(gray, dictionary, board, K, dist))
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
                    cv.aruco.drawAxis(frame, K, dist, rvec, tvec, 0.05)
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
