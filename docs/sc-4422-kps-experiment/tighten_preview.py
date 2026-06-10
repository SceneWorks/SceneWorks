import os, glob, json, numpy as np
from PIL import Image, ImageDraw

FOLDER = "/Users/michael/Library/Application Support/SceneWorks/data/projects/ab.sceneworks/assets/images/genset_e6b07eb5b5374627af1bf47083bac305"
ANGLES = ["front","three_quarter_left","three_quarter_right","left_profile","right_profile",
          "up","down","up_left","up_right","down_left","down_right"]
kps = json.load(open("/tmp/extracted_kps.json"))
pngs = sorted(glob.glob(os.path.join(FOLDER, "*_00*.png")))

# Target: head occupies ~62% of frame height, top margin ~5% -> head-and-shoulders fills frame.
HEAD_FRAC = 0.62
TOP_MARGIN = 0.05

tight = {}
cells = []
for path, angle in zip(pngs, ANGLES):
    k = np.array(kps[angle])  # 5x2 normalized [Leye,Reye,nose,Lmouth,Rmouth]
    eye_y = k[:2,1].mean(); eye_cx = k[:2,0].mean()
    mouth_y = k[3:,1].mean()
    nose_x = k[2,0]
    eyemouth = max(mouth_y - eye_y, 0.04)
    head_top = eye_y - 1.5*eyemouth
    chin     = mouth_y + 0.7*eyemouth
    head_h   = chin - head_top
    # horizontal head center: face centroid pulled toward frame center (esp. profiles)
    cx = 0.6*((eye_cx+nose_x)/2) + 0.4*0.5
    crop = head_h / HEAD_FRAC            # square crop side (normalized)
    top  = head_top - TOP_MARGIN*crop
    left = cx - crop/2
    # clamp into [0,1]
    crop = min(crop, 1.0)
    left = min(max(left, 0.0), 1.0-crop)
    top  = min(max(top, 0.0), 1.0-crop)
    # tightened kps in the NEW frame
    nk = (k - np.array([left, top])) / crop
    tight[angle] = nk.tolist()
    # build preview: crop+resize the reference to show the resulting framing
    img = Image.open(path).convert("RGB"); W,H = img.size
    box = (int(left*W), int(top*H), int((left+crop)*W), int((top+crop)*H))
    cell = img.crop(box).resize((360,360))
    d = ImageDraw.Draw(cell)
    for (x,y) in nk:
        d.ellipse([x*360-4,y*360-4,x*360+4,y*360+4], fill=(0,255,80))
    d.text((6,4), angle, fill=(255,255,0))
    cells.append(cell)
    print(f"{angle:<20} new iocular {abs(nk[1,0]-nk[0,0]):.3f}  eyeY {nk[:2,1].mean():.3f}  (crop {crop:.2f} of frame)")

# montage 4x3
cols, rows = 4, 3
mont = Image.new("RGB",(cols*360, rows*360),(30,30,30))
for i,c in enumerate(cells):
    mont.paste(c, ((i%cols)*360,(i//cols)*360))
mont.save("/tmp/tightened_montage.png")
json.dump(tight, open("/tmp/tightened_kps.json","w"), indent=2)
print("\nsaved /tmp/tightened_montage.png and /tmp/tightened_kps.json")
print("target: iocular ~0.20-0.23, eyeY ~0.33 (vs current InstantID 0.130 / 0.520, vs Qwen 0.165 / 0.346)")
