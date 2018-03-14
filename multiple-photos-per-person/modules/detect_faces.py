import os
import imghdr
from models import Image


def find_images(path):
    ''' finds files in a given path name and returns a list of cv images'''
    if os.path.isdir(path):
        files = os.listdir(path)
        images = []
        for file in files:
            file_path = os.path.join(path, file)
            if imghdr.what(file_path):
                images.append(file_path)
        return images


def find_image(path):
    if imghdr.what(path):
        return path


def show_or_return(path):
    i = Image(path)
    if show:
        i.show_image()
    else:
        i.detect_face()
        return i.face_matrix, i.rect


def main(path=".", show=False):
    if os.path.isdir(path):
        image_paths = find_images(path)
        for path in image_paths:
            return show_or_return(path)
    else:
        if find_image(path):
            return show_or_return(path)


# arguments: first: relative directory path, second: show detected
if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    show = False
    if len(sys.argv) > 2:
        show = sys.argv[2]
    main(path, show)
