import cv2
import numpy as np
import datetime as dt


# ====================================================================

def read_hdr(file_path):
    """Reads the hdr file and splits the data at "=" using the left side of the equal sign as key in a dictionary
    and the right side as value. Hard coded special treatment for the wavelengths since they're two rows."""

    file_object = open(file_path, 'r')
    hdr_dict = {}
    flag = 0
    for line in file_object:
        temp_list = line.split(' = ')
        if len(temp_list) >= 2 or flag == 1:
            if flag == 1:
                hdr_dict["wavelength"] = hdr_dict["wavelength"] + line
                flag = 0
            else:
                hdr_dict[temp_list[0]] = temp_list[1]

            if temp_list[0] == "wavelength":
                flag = 1

    return hdr_dict


# ====================================================================

def resize_img(image, scale_percent):
    """ Resizes an image based on percentage of input image. """

    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(image, dim)

    return resized_img


# ====================================================================

def str_to_array(string):
    """ Turns a string of numbers separated by "," or "space" into an array of those numbers. """

    line = string.replace(',', '')
    temp_list = line.split(' ')
    array = []
    for value in temp_list:
        try:
            int(value)
        except:
            pass
        else:
            array = np.append(array, int(value))

    return array


# ====================================================================

def load_cube(path, number, starting_cube=1, scale=100, normalize=True):
    """Loads and reshapes the data in a cube ".img" file. Also normalises the cube to intensities from 0-255.
    Possibility to also normalise all cubes in an image series if a normalising factor has been
    calculated beforehand."""

    fid = open(str(path + number + ".img"), 'rt')
    data = np.fromfile(fid, dtype=np.float32)
    fid.close()
    hdr_dict = read_hdr(str(path + number + ".hdr"))
    columns = int(hdr_dict.get("samples"))
    rows = int(hdr_dict.get("lines"))
    bands = int(hdr_dict.get("bands"))
    wavelength = str_to_array(hdr_dict.get("wavelength"))

    # Shaping and normalizing cube
    cube = data.reshape(bands,rows, columns)
    cube = np.moveaxis(cube, 0, 2)
    #print(cube.shape)
    cube = np.divide(cube, 2 ** 16)

    # change to image intensity from 0-255 for easier handling
    cube = 255 * cube

    if not scale == 100:
        cube = resize_img(cube, scale)
    if normalize:
        # Loads a precalculated normalising factor.
        normalize_factor = np.load("./Data/Numpy/normalize_factor.npy")
        cube = np.multiply(cube, normalize_factor[:, int(number) - starting_cube])

    return cube, wavelength


# ====================================================================

def save(name, numpy_data=None, cv_data=None):
    """Saves numpy arrays ".npy" and images ".png", with a timestamp."""

    filename = dt.datetime.now()
    filename = filename.strftime("%b-%d_%H%M")
    filename = name + "_" + filename
    if numpy_data is not None:
        np.save(str("./Data/Numpy/" + filename), numpy_data)
    elif cv_data is not None:
        cv2.imwrite(str("./Data/Images/" + filename + ".png"), cv_data)


# ====================================================================

def normalize_cubes(path, starting_cube, no_of_cubes):
    """Used to calculate the normalising factor for each cube. Opens each cube and the same area in each of the images
    has to be selected manually to normalise them. Run once to create the normalising factor."""

    global point_pos
    global area
    area = 10
    print("1", "/", no_of_cubes)
    cube, _ = load_cube(path, str(starting_cube), normalize=False)
    cv2.namedWindow("pick white")
    cv2.setMouseCallback("pick white", _mouse_callback, np.uint8(cube[:, :, 24]))
    cv2.imshow("pick white", np.uint8(cube[:, :, 24]))

    while True:
        key = cv2.waitKey(0)
        x = point_pos[0]
        y = point_pos[1]
        if key == ord("s"):
            start_point = (x - area, y - area)
            end_point = (x + area, y + area)
            save_image = np.uint8(cube[:, :, 24])
            save_image = cv2.rectangle(save_image, start_point, end_point, 255, 1)
            save("normalize", cv_data=save_image)
        else:
            break

    goal_sum = np.sum(cube[y-area:y+area, x-area:x+area, :], axis=(0, 1))
    normalize_factor = np.ones([goal_sum.shape[0], 1])

    # Loops through all cubes, prompts you to select the same area in each cube.
    for index in range(int(starting_cube)+1, no_of_cubes):
        print(index, "/", no_of_cubes)
        cube, _ = load_cube(path, str(index), normalize=False)
        cv2.setMouseCallback("pick white", _mouse_callback, np.uint8(cube[:, :, 24]))
        cv2.imshow("pick white", np.uint8(cube[:, :, 24]))
        while True:
            key = cv2.waitKey(0)
            x = point_pos[0]
            y = point_pos[1]
            if key == ord("s"):
                start_point = (x-area, y-area)
                end_point = (x+area, y+area)
                save_image = np.uint8(cube[:, :, 24])
                save_image = cv2.rectangle(save_image, start_point, end_point, 255, 1)
                save("normalize", cv_data=save_image)
            else:
                break

        cube_sum = np.sum(cube[y-area:y+area, x-area:x+area, :], axis=(0, 1))
        factor = np.divide(goal_sum, cube_sum)
        factor = np.expand_dims(factor, axis=1)
        normalize_factor = np.append(normalize_factor, factor, axis=1)
    save("normalize_factor", numpy_data=normalize_factor)
    cv2.destroyWindow("pick white")


# ====================================================================

def _mouse_callback(event, x, y, param):
    """Mouse interrupt function to save clicked area in the image."""

    global point_pos
    global area
    
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x-area, y-area)
        end_point = (x+area, y+area)
        cv2.rectangle(param, start_point, end_point, 255, 1)
        cv2.imshow("pick white", param)
        point_pos = np.array([x, y])
