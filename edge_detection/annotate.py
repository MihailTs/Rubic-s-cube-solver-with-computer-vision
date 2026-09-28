import os
import cv2
import pandas as pd

# path to raw dataset
PATH = "edge_detection/data"


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    df = param["df"]
    row = param["row"]
    angle = param["angle_num"]
    height = param["height"]
    width = param["width"]

    match angle:
        case 0:
            df.loc[row, "top_left_x"] = x / width
            df.loc[row, "top_left_y"] = y / height
        case 1:
            df.loc[row, "top_right_x"] = x / width
            df.loc[row, "top_right_y"] = y / height
        case 2:
            df.loc[row, "bottom_left_x"] = x / width
            df.loc[row, "bottom_left_y"] = y / height
        case 3:
            df.loc[row, "bottom_right_x"] = x / width
            df.loc[row, "bottom_right_y"] = y / height

    param["angle_num"] += 1


def main():
    image_files = [
        f
        for f in sorted(os.listdir(PATH))
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".jfif"))
    ]

    df = pd.DataFrame(
        columns=[
            "image_name",
            "top_left_x",
            "top_left_y",
            "top_right_x",
            "top_right_y",
            "bottom_left_x",
            "bottom_left_y",
            "bottom_right_x",
            "bottom_right_y",
        ]
    )

    for f in image_files:
        image_path = os.path.join(PATH, f)
        curr_image = cv2.imread(image_path,)

        if curr_image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # Create one row for this image
        row = len(df)
        df.loc[row] = {"image_name": f}
        height, width = curr_image.shape[:2]

        window_name = "Image Sampler - Click 4 corners"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        callback_data = {
            "df": df,
            "row": row,
            "angle_num": 0,
            "width": width,
            "height": height
        }

        cv2.setMouseCallback(window_name, on_mouse, callback_data)

        while callback_data["angle_num"] < 4:
            cv2.imshow(window_name, curr_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyWindow(window_name)
                print(df)
                return

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print(df)
                return

        cv2.destroyWindow(window_name)

    df.to_csv("cube_corners.csv")


if __name__ == "__main__":
    main()