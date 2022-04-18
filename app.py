# from turtle import width
from google.cloud import vision
from PIL import Image, ImageDraw
import argparse
import streamlit as st
import os

def load_image(image_file):
	img = Image.open(image_file)
	return img

def detect_face(face_file, max_results=4):
    client = vision.ImageAnnotatorClient()

    content = face_file.read()
    image = vision.Image(content=content)

    return client.face_detection(
        image=image, max_results=max_results).face_annotations

def highlight_faces(image, faces, output_filename):
    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    # Sepecify the font-family and the font-size
    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        # Place the confidence value/score of the detected faces above the
        # detection box in the output image
        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y - 30),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FF0000')
    im.save(output_filename)

def main(input_filename, output_filename, max_results):
    with open(input_filename, 'rb') as image:
        faces = detect_face(image, max_results)
        print('Found {} face{}'.format(
            len(faces), '' if len(faces) == 1 else 's'))

        print('Writing to file {}'.format(output_filename))
        # Reset the file pointer, so we can read the file again
        image.seek(0)
        highlight_faces(image, faces, output_filename)

if __name__ == '__main__':
    st.subheader("Image")
    image_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    
    input_dir = "/app/images/"
    output_dir = input_dir+"detected/"

    output_file = output_dir+"output.jpg"

    if image_file is not None:
        img = load_image(image_file)
        st.image(img,width=400,caption="The original image")
        with open(os.path.join(input_dir,image_file.name),"wb") as f: 
            f.write(image_file.getbuffer())

        main(input_dir+image_file.name, output_file, 15)
        st.image(load_image(output_file),width=400,caption="Cloud Vision AI's detected image")