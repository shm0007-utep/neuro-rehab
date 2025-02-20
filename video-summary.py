
# Importing all necessary libraries 
import cv2 
import os 
import base64
import requests
import os
import sys
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv('API_KEY')

def encode_image_file(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def encode_frame(frame):
	return base64.b64encode(frame).decode('utf-8')

image_path = "frame0.jpg"

# Getting the base64 string
base64_image = ""

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

responses = []
messages = []
# messages = ["The image depicts a scene where a young boy is looking through a slightly open door, and there is an adult figure on the other side of the door, partially obscured. The boy appears to have a thoughtful or cautious expression, while the adult's face is not fully visible. The setting has a simple, domestic appearance with a plain wall and a stereotypical door.",
# 			 "I can't identify the content of the image, but it appears to depict a scene with a person showing emotion, possibly in a tense or dramatic situation. If you have specific questions or need analysis about the scene, feel free to ask!", 
# 			 "I can't identify specific images or their contents, but it appears to show a person with a distressed or emotional expression. If you describe the context or content, I can help with that!", "I'm unable to identify or describe the specific contents of the image. If you have any questions or need information on a related topic, feel free to ask!", 
# 			 'The image depicts a boy sitting on a bed, wearing headphones and looking contemplative or focused. He is dressed casually, with a striped shirt and a hoodie. In the background, there is a nightstand or dresser with a framed photo and some books. The overall atmosphere appears somewhat subdued or reflective.', 
# 			 "The image depicts a room, likely a child's bedroom, with blue walls and various personal items. There is a boy facing towards the left, and a woman standing in the doorway. The room features a bulletin board filled with notes and photographs, a baseball glove hanging on the wall, a trophy, and a water bottle. The decor includes star shapes on the wall and a framed picture on a surface, likely a dresser. The atmosphere suggests a casual, homely setting.", 
# 			 "I can't tell what's in the image. It appears to show a person holding a dark box or container, but without more details or context, I can't provide specific information about it.",
# 			   "The image shows a child looking at a wooden object or a toy in a dimly lit environment. The child appears contemplative, and the background suggests a setting with shelves and possibly other items, though details remain somewhat obscured due to the lighting.",
#                  'The image appears to depict an intense or dramatic scene, possibly from a performance or a film. A person is shown with a strong expression, possibly in the midst of an action sequence, surrounded by shards of light or reflective fragments. The setting has a warm color tone, suggesting an energetic or emotional atmosphere. If you need specific information about its context or significance, please provide more details!', 'The image appears to depict two individuals, a man and a girl, engaged in a moment of interaction or contemplation, set against a softly blurred background that suggests a natural environment. The lighting and focus emphasize their profiles, giving the scene an emotional or dramatic tone.',
# 				 "The image features a quote attributed to Winston Churchill, which reads:\n\n\"ALL GREAT THINGS ARE SIMPLE,  \nAND MANY CAN BE EXPRESSED IN SINGLE WORDS:  \nFREEDOM, JUSTICE, HONOR, DUTY, MERCY, HOPE.\"\n\nThe background depicts a sunset with silhouettes of several people, possibly a band, standing together, creating a reflective or inspirational atmosphere."]


cam = cv2.VideoCapture("data/dummy.mp4") 

try: 
	
	if not os.path.exists('output'): 
		os.makedirs('output') 

# if not created then raise error 
except OSError: 
	print ('Error: Creating directory of data') 

# frame 
currentframe = 0
fps = cam.get(cv2.CAP_PROP_FPS)
frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
total_count = 10
if len(sys.argv)>1:
	total_count = int(sys.argv[1])
print(f"fps {fps} frame count {frame_count}")
while(currentframe <  frame_count): 
	
	# reading from frame 
	ret,frame = cam.read() 

	if ret : 
		currentframe += 1

		if currentframe % int(frame_count / total_count) != 0:
			continue

		name = './cv' + str(currentframe) + '.jpg'
		print ('Creating...' + name) 
		_, buffer = cv2.imencode('.jpg', frame)
		base64_image = base64.b64encode(buffer).decode('utf-8')
		payload = {
			"model": "gpt-4o-mini",
			"messages": [
				{
				"role": "user",
				"content": [
					{
					"type": "text",
					"text": "What’s in this image? Explain in detail."
					},
					{
					"type": "image_url",
					"image_url": {
						"url": f"data:image/jpeg;base64,{base64_image}"
					}
					}
				]
				}
			],
			"max_tokens": 300
		}
		cv2.imwrite(name, frame) 
		response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
		response_json = response.json()
		responses.append(response_json)
		print(f"Response for {currentframe}")
		print(response_json["choices"][0]["message"]["content"])
		messages.append(response_json["choices"][0]["message"]["content"] )

	else: 
		break

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 
summary_prompt = "I have descriptions of some screenshots of a video at some intervals in cronological order. Surmmarize them in a single paragraph to give me an idea about the video. Desctiptions are: "+ ' '.join(f'{i+1}. {item}' for i, item in enumerate(messages))
print(summary_prompt)
summary_payload = {
			"model": "gpt-4o-mini",
			"messages": [
				{
				"role": "user",
				"content": [
					{
					"type": "text",
					"text": summary_prompt
					}
				]
				}
			],
			"max_tokens": 300
		}
response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=summary_payload)
response_json = response.json()
print(response_json)


