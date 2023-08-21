import cv2
import numpy as np
from keras.models import load_model
import json
# from matplotlib import pyplot as plt
save_video = True
if save_video:
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
facePath = "cascode/haarcascade_frontalface_default.xml"
smilePath = "cascode/haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)
smileCascade = cv2.CascadeClassifier(smilePath)

frame_num = 0
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
model = load_model('weight/weights_original.hdf5')
with open('weight/char_indices.json') as json_file:  
    char_indices = json.load(json_file)
with open('weight/indices_char.json') as json_file:  
    indices_char = json.load(json_file)
# print(indices_char)
# print(char_indices)
maxlen = 40
char_len = 70
seed = "dimon walks into the bar just because el"
def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

generated = ''
sentence = seed[:maxlen]
next_char = None
diversity =0.5
def display_joke(frame):
	global next_char, sentence,generated
	if frame_num%3==0:
		x_pred = np.zeros((1, maxlen, char_len))
		for t, char in enumerate(sentence):
			x_pred[0, t, char_indices[char]] = 1.

		preds = model.predict(x_pred, verbose=0)[0]
		next_index = sample(preds, diversity)
		next_char = indices_char[str(next_index)]

		generated += next_char
		generated = generated[-38:]
		sentence = sentence[1:] + next_char
		cv2.putText(frame,generated,(20,30), font, 1, (0,0,0), 2, cv2.LINE_AA)
		print(f'{diversity}:\n {repr(generated)}')
	else:
		cv2.putText(frame,generated[-40:],(20,30), font, 1, (200,255,155), 1, cv2.LINE_AA)

# display_joke(5)

while(cap.isOpened()):
	ret, frame = cap.read()

	if ret == True:
		smile_detected = False
		frame = cv2.flip( frame, 1 )
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		faces = faceCascade.detectMultiScale(gray, scaleFactor= 1.05,
			minNeighbors=8, minSize=(55, 55), flags=cv2.CASCADE_SCALE_IMAGE)

		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = frame[y:y+h, x:x+w]

			smile = smileCascade.detectMultiScale(
				roi_gray,
				scaleFactor= 1.7,
				minNeighbors=22,
				minSize=(25, 25),
				flags=cv2.CASCADE_SCALE_IMAGE
				)
			if len(smile) > 0:
				smile_detected = True
			for (x, y, w, h) in smile:
				# print('smile')
				cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
		if not smile_detected and len(faces)>0:
			display_joke(frame)
		elif len(faces)==0:
			pass
		else:
			generated = ''
		cv2.imshow('frame',frame)
		if save_video:
			out.write(frame)
	else:
		break
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



cap.release()
cv2.destroyAllWindows()
if save_video:
	out.release()