import face_recognition
from PIL import Image, ImageDraw
import cv2

cam = cv2.VideoCapture(0)
s, Image_OnClick = cam.read()

cv2.imwrite("test.jpg",Image_OnClick)

yash_image = face_recognition.load_image_file("Yash.jpg")
yash_face_encoding = face_recognition.face_encodings(yash_image)[0]

sumedh_image = face_recognition.load_image_file("Sumedh.jpeg")
sumedh_face_encoding = face_recognition.face_encodings(sumedh_image)[0]

yashom_image = face_recognition.load_image_file("Yashom.jpg")
yashom_face_encoding = face_recognition.face_encodings(yashom_image)[0]


known_face_encodings = [
    yash_face_encoding,
    sumedh_face_encoding,
    yashom_face_encoding,
]
known_face_names = [
    "Yash Turkar",
    "Sumedh Deshpande",
    "Yashom Dighe",
]

unknown_image = face_recognition.load_image_file("four.jpg")

face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)



pil_image = Image.fromarray(unknown_image)

draw = ImageDraw.Draw(pil_image)


for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    matches = face_recognition.compare_faces(known_face_encodings, face_encoding , tolerance=0.5)

    name = "Unknown"


    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]


    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))


    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))



del draw


pil_image.show()



