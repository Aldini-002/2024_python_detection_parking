import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
import os


def deteksi_parkir(sumber_video):
    cap = cv2.VideoCapture(int(sumber_video) if sumber_video.isdigit() else sumber_video)
    if not cap.isOpened():
        messagebox.showerror('','video tidak bisa di buka')
        return

    points = np.array([], dtype=np.int32)
    drawing = False
    modal_active = False
    spaces = []

    root.withdraw()

    def mouse_event(event, x, y, flags, params):
        nonlocal points, drawing, spaces, modal_active
        if modal_active:
            return

        space_names_in_spaces = {space["space_name"] for space in spaces}

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = np.array([[x, y]], dtype=np.int32)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                coordinate = np.array([[x, y]], dtype=np.int32)
                points = np.vstack([points, coordinate])
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing:
                drawing = False
                modal_active = True
                current_name = simpledialog.askstring("", "Nama area")
                if current_name:
                    current_name = current_name.upper()
                    if (current_name in space_names_in_spaces):
                        messagebox.showwarning(f"{current_name} sudah digunakan")
                    else:
                        spaces.append({"space_name": current_name, "polylines": points.tolist()})
                modal_active = False
                points = np.array([], dtype=np.int32)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not drawing:
                for space in spaces:
                    target = cv2.pointPolygonTest(
                        np.array([space["polylines"]], dtype=np.int32), (x, y), False
                    )
                    if target >= 0:
                        spaces = [item for item in spaces if item["space_name"] != space["space_name"]]

    model = YOLO(os.path.join("data", "model", "yolov8s.pt"))

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = cv2.resize(frame, (720, 480))

        results = model(frame)
        cars = []
        space_names = [space["space_name"] for space in spaces]
        empty_space = space_names
        filled_space = []

        if not drawing:
            for result in results:
                for box in result.boxes:
                    cx, cy, w, h = box.xywh[0].cpu().numpy().tolist()
                    cx, cy, w, h = int(cx), int(cy), int(w), int(h)
                    confidence = box.conf.cpu().item()
                    class_id = box.cls.cpu().item()
                    class_name = model.names[class_id]

                    if class_name == "car" or class_name == "truck":
                        cars.append([cx, cy])

        for space in spaces:
            polylines = np.array([space["polylines"]], dtype=np.int32)
            cv2.polylines(
                frame, [polylines], isClosed=True, color=(255, 255, 0), thickness=2
            )

            centroid = np.mean(polylines[0], axis=0).astype(int)
            centroid = (centroid[0], centroid[1])

            text = space["space_name"]
            font_scale = 0.5
            font_thickness = 1
            text_size, _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            text_x = centroid[0] - text_size[0] // 2
            text_y = centroid[1] + text_size[1] // 2

            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 0),
                font_thickness,
                cv2.LINE_AA,
            )

            for car in cars:
                cx = car[0]
                cy = car[1]

                target = cv2.pointPolygonTest(polylines, (cx, cy), False)
                if target >= 0:
                    cv2.polylines(
                        frame,
                        [polylines],
                        isClosed=True,
                        color=(255, 0, 255),
                        thickness=2,
                    )
                    filled_space.append(space["space_name"])
                    empty_space = [
                        item for item in space_names if item not in filled_space
                    ]

        if drawing and points.size > 0:
            cv2.polylines(
                frame, [points], isClosed=True, color=(255, 255, 0), thickness=2
            )

        cv2.putText(
            frame,
            f"Parkir : {space_names}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Kosong : {empty_space}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Terisi : {filled_space}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        cv2.imshow('Deteksi parkir', frame)
        cv2.setMouseCallback('Deteksi parkir', mouse_event)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.deiconify()


root = tk.Tk()
root.title("Deteksi parkir")


label_sumber_video = tk.Label(root, text="Sumber vidoe:")
label_sumber_video.grid(column=0, row=0, padx=10, pady=10)

entry_sumber_video = tk.Entry(root)
entry_sumber_video.grid(column=1, row=0, padx=10, pady=10)


button_mulai = tk.Button(root, text="Mulai", command=lambda: deteksi_parkir(entry_sumber_video.get()))
button_mulai.grid(column=0, row=2, columnspan=2, padx=10, pady=10)


root.mainloop()