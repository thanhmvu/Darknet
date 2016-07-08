[Generate Labels for VOC]

Now we need to generate the label files that Darknet uses. Darknet wants a .txt file for each image with a line for each ground truth object in the image that looks like:

<object-class> <x> <y> <width> <height>

Where x, y, width, and height are relative to the image's width and height, with x and y are the location of the center of the object:

    x = x_center / img_w
    y = y_center / img_h
    with = obj_w / img w
    height = obj_h / img_h
