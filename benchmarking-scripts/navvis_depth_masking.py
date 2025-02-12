import cv2
import os
import numpy as np
DEBUG = True

def float32_to_int16_depth(depth_map, max_depth=None):
  """
  Converts a float32 depth map to an int16 depth map.

  Args:
      depth_map: A numpy array representing the float32 depth map.
      max_depth: The maximum valid depth value (optional). Defaults to None.

  Returns:
      A numpy array representing the int16 depth map.
  """

  # Handle potential NaNs (Not a Number)
  depth_map = np.nan_to_num(depth_map)

  # Clip values to a valid range (optional)
  if max_depth is not None:
    depth_map = np.clip(depth_map, 0, max_depth)

  # Scaling factor (adjust based on your depth map range)
  scale = np.iinfo(np.int16).max / depth_map.max()

  # Convert to int16 with scaling and rounding
  int16_depth = np.round(depth_map * scale).astype(np.int16)

  return int16_depth

def mask_imagexx(image, depth):
    """
    Crop the image to the boundary of the depth map
    """
    depth = float32_to_int16_depth(depth)
    _, binary_mask = cv2.threshold(depth, 1, 255, cv2.THRESH_BINARY)

    # Find the contours of the white region in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("binary_mask", binary_mask)
    cv2.waitKey(0)
    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(contours[0])
    print(x, y, w, h)
    # Crop the image to the bounding box
    cropped_image = image[y:y+h, x:x+w]

    if DEBUG:
        cv2.imshow("Binary Mask", binary_mask)
        #cv2.imshow("Masked Image", cropped_image)
        cv2.waitKey(0)


def mask_image(image, depth):
    """
    Mask the image to the depth map where values are present
    """
    if DEBUG:
        cv2.imshow("Original", float32_to_int16_depth(depth))
        cv2.waitKey(0)
    # First dilate the depthmap
    kernel = np.ones((3, 3), np.uint8)
    depth = cv2.dilate(depth, kernel, iterations=1)

    # Mask the the image to the size of the depth map
    masked_image = np.zeros_like(image)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i, j] != 0:
                masked_image[i, j] = image[i, j]

    if DEBUG:
        cv2.imshow("Dilated Depth", float32_to_int16_depth(depth))
        cv2.imshow("Masked Image", masked_image)
        cv2.waitKey(0)
    return masked_image



def resize_depth_preserve(depth, shape):
    """
    https://github.com/TRI-ML/packnet-sfm/blob/f59b1d615777a9987285a10e45b5d87b0369fa7d/packnet_sfm/datasets/augmentations.py#L56

    Resizes depth map preserving all valid depth pixels
    Multiple downsampled points can be assigned to the same pixel.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W,1]
        Resized depth map
    """

    def is_tuple(data):
        """Checks if data is a tuple."""
        return isinstance(data, tuple)

    def is_list(data):
        """Checks if data is a list."""
        return isinstance(data, list)

    def is_seq(data):
        """Checks if data is a list or tuple."""
        return is_tuple(data) or is_list(data)

    # Return if depth value is None
    if depth is None:
        return depth
    # If a single number is provided, use resize ratio
    if not is_seq(shape):
        shape = tuple(int(s * shape) for s in depth.shape)
    # Store dimensions and reshapes to single column
    depth = np.squeeze(depth)
    h, w = depth.shape
    x = depth.reshape(-1)
    # Create coordinate grid
    uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
    # Filters valid points
    idx = x > 0
    crd, val = uv[idx], x[idx]
    # Downsamples coordinates
    crd[:, 0] = (crd[:, 0] * (shape[0] / h)).astype(np.int32)
    crd[:, 1] = (crd[:, 1] * (shape[1] / w)).astype(np.int32)
    # Filters points inside image
    idx = (crd[:, 0] < shape[0]) & (crd[:, 1] < shape[1])
    crd, val = crd[idx], val[idx]
    # Creates downsampled depth image and assigns points
    depth = np.zeros(shape)
    depth[crd[:, 0], crd[:, 1]] = val
    # Return resized depth map
    ret_depth = np.expand_dims(depth, axis=2)

    ret_depth = np.float32(ret_depth)
    return ret_depth


def remove_invalid_depth_edges(depth, image):
    removed_area_str = ""
    # Columns
    validity_mask = depth > 0.01
    validity_mask = validity_mask.astype("uint8")

    column_mask = validity_mask.sum(axis=0)
    column_mask = np.array(column_mask, dtype=bool)

    left_black = next((idx for idx, x in enumerate(column_mask) if x == True), 0)
    right_black = next(
        (len(column_mask) - idx for idx, x in enumerate(reversed(column_mask)) if x == True), len(column_mask)
    )
    # print("left_black", left_black)
    # print("right_black", right_black)
    # print(column_mask[left_black-1], column_mask[left_black], column_mask[left_black+1], column_mask[left_black+2], column_mask[left_black+3], column_mask[left_black+4], column_mask[left_black+5], column_mask[left_black+6])
    column_mask[left_black:right_black] = True
    removed_area_str += f"__x_{left_black}_{right_black}"

    depth = depth[:, column_mask]
    if image is not None:
        image = image[:, column_mask]

    # Rows
    validity_mask = depth > 0.001
    validity_mask = validity_mask.astype("uint8")

    row_mask = validity_mask.sum(axis=1)
    row_mask = np.array(row_mask, dtype=bool)

    # TODO: Do something about off chance lines are deleted in middle of image???
    # print("row_mask", row_mask)
    # first = next(elm for elm in row_mask if elm == False)
    # print("first", first)

    top_black = next((idx for idx, x in enumerate(row_mask) if x == True), 0)
    bottom_black = next((len(row_mask) - idx for idx, x in enumerate(reversed(row_mask)) if x == True), len(row_mask))
    # print(top_black, bottom_black)
    row_mask[top_black:bottom_black] = True
    removed_area_str += f"__y_{top_black}_{bottom_black}"

    depth = depth[row_mask, :]
    if image is not None:
        image = image[row_mask, :]

    return depth, image, removed_area_str


def resize_navvis(depth, image):
    # Not sure which way of resizing works best - mde is going to be trained on 640 x 640
    print("General Resize:")
    reference_px = 420
    h, w = image.shape[:2]
    if h > reference_px * 4 and w > reference_px * 4:
        print("\t resize by 1/4. original shape = ", image.shape)
        result_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
   
    elif h > reference_px * 2 and w > reference_px * 2:
        print("\t resize by 1/2. original shape = ", image.shape)
        result_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
     
    else:
        print("\t resize NOT", image.shape)
        result_image = image
      
    """
        Resize of sparse depth leads to many 0 values becoming > eps 
    """
    if depth.shape[:2] == result_image.shape[:2]:
        result_depth = depth
    else:
        result_depth = resize_depth_preserve(depth, result_image.shape[:2])
        #result_depth = cv2.resize(depth, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    print(f"\t Reduced image size from {image.shape} to {result_image.shape}")
    print("\t depth", np.min(result_depth), np.max(result_depth), np.mean(result_depth))
    image = result_image
    depth = result_depth

    return result_image, result_depth

    
    print("Rotate")
    image, rot_direction = NavvisMDEDataSet.rotate(image_path, image)
    image = np.clip(image, 0, 255)
    # WELL, we should have a rotate-depth-preserve !!!
    depth, _ = NavvisMDEDataSet.rotate(image_path, depth)
    depth = np.clip(depth, 0.0, 10000.0)
    name_ext += f"__{rot_direction}"
    print("\t depth", np.min(depth), np.max(depth), np.mean(depth))

    if do_inpainting is True:
        print(f"Inpaint: {depth.shape}")
        depth = NavvisMDEDataSet.inpaint(depth)
        print("\t depth", np.min(depth), np.max(depth), np.mean(depth))

        print("Clip")
        min_depth_in_ground_truth = 0.5
        depth = np.clip(depth, min_depth_in_ground_truth, 10000.0)
        print("\t depth", np.min(depth), np.max(depth), np.mean(depth))
    else:
        # Rotation makes zeros into small depth values which should be reversed
        epsilon = 0.1
        depth[depth < epsilon] = 0.0

def image_resize_in_ratio(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized