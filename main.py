import ast
import numpy as np
import time
from PIL import Image


data_dir = "./data/"

with open(data_dir + "settings.txt") as f:
    settings = f.read()
    settings = ast.literal_eval(settings)

with open(
    "./wavelengths"
) as f:  # For code purity removed "wavelengths = " from the file
    wavelengths = f.read()
    wavelengths = ast.literal_eval(wavelengths)


def find_spectrum_idx(start: float, end: float, data: list) -> list:
    """Finds first and last index of desired spectrum in given array

    Args:
        start (float): first wavelength of needed spectrum
        end (float): last wavelength of needed spectrum
        data (list): Must be a list of descending wavelengths(float) that camera registers

    Returns:
        list: [start_idx,end_idx]
    """
    start_idx, end_idx = 0, 0

    for wl_idx in range(len(data)):
        if data[wl_idx] <= start and start_idx == 0:
            start_idx = wl_idx
        if data[wl_idx] <= end and end_idx == 0:
            end_idx = wl_idx - 1

    return [end_idx, start_idx]  # reversed because data sorted as descending


rgb_spectrum = {
    "r": find_spectrum_idx(550, 760, wavelengths),
    "g": find_spectrum_idx(510, 550, wavelengths),
    "b": find_spectrum_idx(450, 490, wavelengths),
}


def main():
    frames = settings["frames"]
    ready_pic = np.ndarray((frames, 2048, 3), "uint8")
    for i in range(frames):
        pic = np.load(data_dir + f"cam_{i}.npy")
        red_slice = pic[rgb_spectrum["r"][0] : rgb_spectrum["r"][1]]

        green_slice = pic[rgb_spectrum["g"][0] : rgb_spectrum["g"][1]]

        blue_slice = pic[rgb_spectrum["b"][0] : rgb_spectrum["b"][1]]
        r = red_slice.mean(0)
        g = green_slice.mean(0)
        b = blue_slice.mean(0)

        ready_pic[i] = np.transpose([r, g, b])
        if i % 500 == 0:
            print(i, " lines done")

    im = Image.fromarray(ready_pic.astype(np.uint8))
    im = im.transpose(Image.TRANSPOSE)

    im.save("result.jpg")


if __name__ == "__main__":

    time_s = time.time()
    main()
    print("Done! Time passed: ", (time.time() - time_s))
