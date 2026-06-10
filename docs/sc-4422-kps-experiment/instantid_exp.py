import os, sys, json, time, importlib
import numpy as np, cv2, torch
from PIL import Image

WORKER = "/Users/michael/Repos/SceneWorks/apps/worker/scene_worker"
VENDOR = os.path.join(WORKER, "_vendor", "instantid")
sys.path.insert(0, VENDOR)
OUT = "/tmp/instantid_exp"; os.makedirs(OUT, exist_ok=True)
REF = "/Users/michael/Library/Application Support/SceneWorks/data/projects/ab.sceneworks/assets/images/genset_e6b07eb5b5374627af1bf47083bac305/2026-06-10_qwen_image_edit_2511_lightning_22-year-old-woman-with-fair-complexion-a-p_0001.png"
SEED = 3004710356
PROMPT = ("22 year old woman, with fair complexion, a petite thin frame. short cropped brown hair with auburn "
          "highlights, slightly upturned nose, neutral expression, plain grey background, sharp focus, photorealistic, 4k")
NEG = ("cropped, multiple people, plastic skin, airbrushed, cgi, 3d render, cartoon, anime, waxy, deformed, blurry")

CURRENT_FRONT = [(0.4460,0.5227),(0.5755,0.5166),(0.5106,0.5947),(0.4653,0.6660),(0.5630,0.6613)]
tight = json.load(open("/tmp/tightened_kps.json"))

print("loading antelopev2...", flush=True)
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="antelopev2", root=os.path.expanduser("~/.insightface"), providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640,640))
bgr = cv2.imread(REF)
faces = app.get(bgr)
face = max(faces, key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
face_emb = face["embedding"]
print("got identity embedding", face_emb.shape, flush=True)

print("loading InstantID pipeline (RealVisXL)...", flush=True)
mod = importlib.import_module("pipeline_stable_diffusion_xl_instantid")
Pipe, draw_kps = mod.StableDiffusionXLInstantIDPipeline, mod.draw_kps
import diffusers
from huggingface_hub import hf_hub_download
dtype = torch.bfloat16
idnet = diffusers.ControlNetModel.from_pretrained("InstantX/InstantID", subfolder="ControlNetModel", torch_dtype=dtype)
try:
    pipe = Pipe.from_pretrained("SG161222/RealVisXL_V5.0", controlnet=idnet, torch_dtype=dtype, variant="fp16")
except Exception:
    pipe = Pipe.from_pretrained("SG161222/RealVisXL_V5.0", controlnet=idnet, torch_dtype=dtype)
pipe.load_ip_adapter_instantid(hf_hub_download("InstantX/InstantID", "ip-adapter.bin"))
pipe.to("mps")
print("pipeline ready", flush=True)

def render(name, kps_norm):
    side = 1024
    kps_px = np.array(kps_norm, dtype=np.float32) * side
    control = draw_kps(Image.new("RGB",(side,side),(0,0,0)), kps_px)
    pipe.set_ip_adapter_scale(0.8)
    g = torch.Generator("cpu").manual_seed(SEED)
    t=time.time()
    img = pipe(prompt=PROMPT, negative_prompt=NEG, image_embeds=face_emb, image=control,
               controlnet_conditioning_scale=0.8, ip_adapter_scale=0.8, width=side, height=side,
               num_inference_steps=30, guidance_scale=3.0, generator=g).images[0].convert("RGB")
    img.save(os.path.join(OUT, name+".png"))
    control.save(os.path.join(OUT, name+"_kps.png"))
    print(f"  rendered {name} in {time.time()-t:.0f}s", flush=True)
    return img

jobs = [("01_front_CURRENT", CURRENT_FRONT),
        ("02_front_TIGHT", tight["front"]),
        ("03_three_quarter_left_TIGHT", tight["three_quarter_left"]),
        ("04_left_profile_TIGHT", tight["left_profile"])]
imgs = []
for name, kps in jobs:
    imgs.append((name, render(name, kps)))

# side-by-side montage
cell=512; m=Image.new("RGB",(cell*len(imgs), cell),(20,20,20))
from PIL import ImageDraw
for i,(name,im) in enumerate(imgs):
    c=im.resize((cell,cell)); ImageDraw.Draw(c).text((8,8), name, fill=(255,255,0)); m.paste(c,(i*cell,0))
m.save(os.path.join(OUT,"montage.png"))
print("DONE ->", os.path.join(OUT,"montage.png"), flush=True)
