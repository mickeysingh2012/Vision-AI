import cv2
from pyzbar import pyzbar
import sys
import pyperclip
import time

def set_focus(camera, value):
    # Try to set manual focus; not all cameras support it
    if not camera.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        print("Warning: Autofocus control not supported.")
    if not camera.set(cv2.CAP_PROP_FOCUS, value):
        print("Warning: Manual focus not supported.")

def scan_barcode():
    camera_index = 0  # try 0 by default, can change to 1 if needed
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        sys.exit(1)

    focus_value = 0.0
    set_focus(cap, focus_value)
    last_focus_change_time = time.time()

    scanned_data = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        barcodes = pyzbar.decode(frame)
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            barcode_data = barcode.data.decode('utf-8')
            barcode_type = barcode.type

            if barcode_data not in scanned_data:
                try:
                    pyperclip.copy(barcode_data)
                except pyperclip.PyperclipException as e:
                    print(f"Barcode: {barcode_data} ({barcode_type})")
                    scanned_data.add(barcode_data)

            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{barcode_data} ({barcode_type})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow('Barcode Scanner', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Change focus every 3 seconds (if supported)
        if time.time() - last_focus_change_time > 3:
            focus_value = (focus_value + 0.1) % 1.0
            set_focus(cap, focus_value)
            last_focus_change_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    scan_barcode()
