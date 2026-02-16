import argparse
import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe camera indices.")
    parser.add_argument("--max-index", type=int, default=10, help="Largest index to test.")
    args = parser.parse_args()

    print("Detecting cameras...")
    found = []
    for idx in range(args.max_index + 1):
        cap = cv2.VideoCapture(idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            found.append((idx, w, h))
            print(f"  index={idx}: OK ({w}x{h})")
        cap.release()

    if not found:
        print("No cameras found.")
        return

    print("\nUse two indices (left/right) in calibrate.py and depth_live.py.")
    print("Found Cameras: ")
    for idx, w, h in found:
        print(f"  index={idx}: {w}x{h}")


if __name__ == "__main__":
    main()
