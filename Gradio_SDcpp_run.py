"""
20250122 - first revision
20250130 - testing subprocess.run to capture stdout
- TOdo catch the log stdOut from the terminal command and log it
- Progress somehow?
"""

import random, string
import gradio as gr
from rich.console import Console
import os
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import subprocess
import io
import time
import subprocess
import sys
from gradio_log import Log  #https://github.com/louis-she/gradio-log


def genRANstring(n):
    """
    n = int number of char to randomize
    Return -> str, the filename with n random alphanumeric charachters
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    print(f'IM-{res}.png  CREATED')
    return f'IM-{res}.png'


console = Console(width=80)
theme=gr.themes.Default(primary_hue="blue", secondary_hue="pink",
                        font=[gr.themes.GoogleFont("Lato"), "Arial", "sans-serif"]) 
SAMPLINGMETHOD = ['euler', 'euler_a', 'heun', 'dpm2', 'dpm++2s_a', 'dpm++2m', 'dpm++2mv2', 'ipndm', 'ipndm_v', 'lcm']

def initiateLOG():
    filename = "test.log"
    with open(filename, 'w') as file:
        # Write some text to the file
        file.write("Start Logging\n")
    file.close()

initiateLOG()

def openDIR():
    """
    Open the current working directory in windows explorer
    """
    import os
    current_directory = os.getcwd()
    print("Current Directory:", current_directory)
    os.system(f'start explorer "{current_directory}"')


############### CREATE IMAGE ##########################
def CreateImage(PROMPT,STEPS,WIDTH,HEIGHT, CONFIGSCALE,SAMPLINGMETHOD):
    """
    Use in terminal sd.exe stable diffusion cpp to create an image from a given prompt, and then it displays it in the gradio app
    PROMPT -> str, with no line brakes.
    STEPS - > int, number of sample steps for the diffuser
    WIDTH,HEIGHT -> int, image dimensions: must be multiples of 64 the bigger the image, the bigger the VRAM requirements and generation time
    CONFIGSCALE -> int, unconditional guidance scale: (default: 7.0)
    SAMPLINGMETHOD -> str from choices: euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, ipndm, ipndm_v, lcm
    Returns:
    targetimage -> PIL object 
    fILENAME -> str, filename of the image
    """
    fILENAME = genRANstring(5)
    modelname = 'dreamshaper_8.safetensors'
    PROMPT = PROMPT.replace('\n','')
    PROMPT = PROMPT.replace('\n\n','')
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
        def isatty(self):
            return False 
    print(f'Generating image with dreamshaper_8 as: {fILENAME}')
    start = datetime.now()
    args = ['sd.exe',
        '-m',
        modelname,
        '--cfg-scale',
        str(CONFIGSCALE),
        '--steps',
        str(STEPS),
        '-W',
        str(WIDTH),
        '-H',
        str(HEIGHT),
        '--sampling-method',
        SAMPLINGMETHOD,
        '-p',
        PROMPT,
        '-o',
        fILENAME]

    initiateLOG()
    filename = "test.log"
    with io.open(filename, "wb") as writer, io.open(filename, "rb", 1) as reader:
        process = subprocess.Popen(args, stdout=writer)
        while process.poll() is None:
            sys.stdout.write(reader.read().decode("utf-8"))
            time.sleep(1)
        # Read the remaining
        sys.stdout.write(reader.read().decode("utf-8"))

    delta = datetime.now() - start
    print(f'Saving the Generated image with dreamshaper_8 as: {fILENAME}')
    print(f'Completed in {delta}')
    targetimage = Image.open(fILENAME)
    return targetimage, fILENAME

with gr.Blocks(fill_width=True,theme=theme) as demo:
    # INTERFACE
    filename = "test.log"
    with gr.Row(variant='panel'):
        gr.HTML(
        f"""<h1 style="text-align:center">Generate images with Stable Diffusion CPP</h1>""")       
    with gr.Row():
        #HYPERPARAMETERS
        with gr.Column(scale=1):
            gr.Markdown('---')
            i_steps = gr.Slider(minimum=1,maximum=30,value=10,step=1,label='Steps')
            i_width = gr.Slider(minimum=64,maximum=960,value=512,step=64,label='Width')
            i_height = gr.Slider(minimum=64,maximum=960,value=256,step=64,label='Height')
            i_confScale = gr.Slider(minimum=1,maximum=30,value=7,step=1,label='Guidance Scale')
            i_samplMethod = gr.Dropdown(choices=SAMPLINGMETHOD,value='euler_a',multiselect=False,container=True)
            GEN_IMAGE = gr.Button(value='Generate Image',variant='primary')
            gr.Markdown('---')
            ImageFilename = gr.Textbox(lines=2,label='Generated Image Filename',show_copy_button=True)
            OPEN_FOLDER = gr.Button(variant='secondary',value='Open Image Folder')
            clear = gr.ClearButton()
        with gr.Column(scale=3):
            SDPrompt = gr.Textbox(lines=8,label='SD PROMPT')
            SDImage = gr.Image(type='pil',label='Generated Image',show_download_button=True, show_fullscreen_button=True,)
    with gr.Row():
            # https://github.com/louis-she/gradio-log/blob/master/backend/gradio_log/log.py
            SDlogs = Log(filename, dark=True, xterm_font_size=12,height=160,
                         every=1,label='Progress') #gr.Textbox(lines=4,label='Progress')        
    GEN_IMAGE.click(CreateImage,[SDPrompt,i_steps,i_width,i_height,i_confScale,i_samplMethod],[SDImage,ImageFilename])
    OPEN_FOLDER.click(openDIR,[],[])


if __name__ == "__main__":
    demo.launch()   

