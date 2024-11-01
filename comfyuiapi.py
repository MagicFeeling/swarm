#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import random

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    current_node = ""
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['prompt_id'] == prompt_id:
                    if data['node'] is None:
                        break #Execution is done
                    else:
                        current_node = data['node']
        else:
            if current_node == 'save_image_websocket_node':
                images_output = output_images.get(current_node, [])
                images_output.append(out[8:])
                output_images[current_node] = images_output

    return output_images

prompt_text = """
{"5": {"inputs": {"width": 1400, "height": 1400, "batch_size": 1}, "class_type": "EmptyLatentImage"}, "6": {"inputs": {"text": "upper body shot of gorgeous white skinned very athletic female warrior, in mountain forest. She is wearing a short very short golden metal armor, revealing her huge thighs. Her massively huge breasts pour out of her dress. insanely detailed| intricate detail| ornate| cinematic lighting| ultra realistic 8k", "clip": ["27", 1]}, "class_type": "CLIPTextEncode"}, "8": {"inputs": {"samples": ["13", 0], "vae": ["10", 0]}, "class_type": "VAEDecode"}, "9": {"inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}, "class_type": "SaveImage"}, "10": {"inputs": {"vae_name": "ae.safetensors"}, "class_type": "VAELoader"}, "11": {"inputs": {"clip_name1": "t5xxl_fp8_e4m3fn.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}, "class_type": "DualCLIPLoader"}, "12": {"inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"}, "13": {"inputs": {"noise": ["25", 0], "guider": ["22", 0], "sampler": ["16", 0], "sigmas": ["17", 0], "latent_image": ["5", 0]}, "class_type": "SamplerCustomAdvanced"}, "16": {"inputs": {"sampler_name": "euler"}, "class_type": "KSamplerSelect"}, "17": {"inputs": {"scheduler": "simple", "steps": 20, "denoise": 1.0, "model": ["12", 0]}, "class_type": "BasicScheduler"}, "22": {"inputs": {"model": ["27", 0], "conditioning": ["6", 0]}, "class_type": "BasicGuider"}, "25": {"inputs": {"noise_seed": 723124458954192}, "class_type": "RandomNoise"}, "26": {"inputs": {"lora_name": "flux-large_breasts-v2_rank16_bf16-step01500.safetensors", "strength_model": 0.76, "strength_clip": 1.0, "model": ["12", 0], "clip": ["11", 0]}, "class_type": "LoraLoader"}, "27": {"inputs": {"lora_name": "Hand v2.safetensors", "strength_model": 1.0, "strength_clip": 1.0, "model": ["28", 0], "clip": ["28", 1]}, "class_type": "LoraLoader"}, "28": {"inputs": {"lora_name": "Hanging_breasts-000023.safetensors", "strength_model": 0.24, "strength_clip": 1.0, "model": ["26", 0], "clip": ["26", 1]}, "class_type": "LoraLoader"}}
"""

prompt = json.loads(prompt_text)
prompt["25"]["inputs"]["noise_seed"] = random.randint(0, 1000000000000)

prompt["26"]["inputs"]["lora_name"]="flux-large_breasts-v2_rank16_bf16-step01500.safetensors"
prompt["26"]["inputs"]["strength_model"]=0.70

prompt["28"]["inputs"]["lora_name"]="Hand v2.safetensors"
prompt["28"]["inputs"]["strength_model"]=1

prompt["27"]["inputs"]["lora_name"]="Hanging_breasts-000023.safetensors"
prompt["27"]["inputs"]["strength_model"]=0.20

prompt["6"]["inputs"]["text"] = "upper body shot of gorgeous white skinned very athletic  female french maid, in luxury apartment. She is wearing a very short french maid dress, revealing her huge thighs. Her massively huge breasts pour out of her dress. insanely detailed| intricate detail| ornate| cinematic lighting| ultra realistic 8k"



ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
images = get_images(ws, prompt)
ws.close() # for in case this example is used in an environment where it will be repeatedly called, like in a Gradio app. otherwise, you'll randomly receive connection timeouts
#Commented out code to display the output images:

# for node_id in images:
#     for image_data in images[node_id]:
#         from PIL import Image
#         import io
#         image = Image.open(io.BytesIO(image_data))
#         image.show()





from swarm import Swarm, Agent

client = Swarm()


def instructions(context_variables):
    name = context_variables.get("name", "User")
    return f"You are a helpful agent. Greet the user by name ({name})."


# printing
def print_account_details(context_variables: dict):
    preference = context_variables.get("preference", None)
    name = context_variables.get("name", None)
    print(f"Preference Details: {preference}")
    return "Success"


agent = Agent(
    name="Agent",
    model="Casual-Autopsy_L3-Umbral-Mind-RP-v1.0-8B",
    instructions=instructions,
    functions=[print_account_details],
)

context_variables = {"name": "Martin", "preference": "Huge breasts"}

response = client.run(
    messages=[{"role": "user", "content": "Hi!"}],
    agent=agent,
    context_variables=context_variables,
)
print(response.messages[-1]["content"])

response = client.run(
    messages=[{"role": "user", "content": "Create me the description of an image with max 100 characters!"}],
    agent=agent,
    context_variables=context_variables,
)
print(response.messages[-1]["content"])