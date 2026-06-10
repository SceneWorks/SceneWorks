import os, glob, json, numpy as np, cv2
from insightface.app import FaceAnalysis

FOLDER = "/Users/michael/Library/Application Support/SceneWorks/data/projects/ab.sceneworks/assets/images/genset_e6b07eb5b5374627af1bf47083bac305"
ANGLES = ["front","three_quarter_left","three_quarter_right","left_profile","right_profile",
          "up","down","up_left","up_right","down_left","down_right"]

app = FaceAnalysis(name="antelopev2", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640,640))

pngs = sorted(glob.glob(os.path.join(FOLDER, "*_00*.png")))
assert len(pngs) == 11, f"expected 11, got {len(pngs)}"

out = {}
print(f"{'angle':<20} {'det':<4} {'iocular':<8} {'eyeY':<6} {'faceH%':<7} {'cx':<6} {'cy':<6}  kps(normalized 0-1, [Leye,Reye,nose,Lmouth,Rmouth])")
for path, angle in zip(pngs, ANGLES):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    faces = app.get(img)
    if not faces:
        print(f"{angle:<20} {'NO':<4} (no face detected)")
        out[angle] = None
        continue
    f = max(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    kps = f.kps.astype(float)  # 5x2 pixels
    nk = kps / np.array([w, h])
    x1,y1,x2,y2 = f.bbox
    iocular = abs(nk[1,0]-nk[0,0])           # eye-to-eye x distance (normalized)
    eyeY = (nk[0,1]+nk[1,1])/2
    faceH = (y2-y1)/h
    cx = nk[:,0].mean(); cy = nk[:,1].mean()
    out[angle] = nk.tolist()
    kps_str = " ".join(f"({p[0]:.3f},{p[1]:.3f})" for p in nk)
    print(f"{angle:<20} {'yes':<4} {iocular:<8.3f} {eyeY:<6.3f} {faceH*100:<7.1f} {cx:<6.3f} {cy:<6.3f}  {kps_str}")

with open("/tmp/extracted_kps.json","w") as fp:
    json.dump(out, fp, indent=2)
print("\nsaved /tmp/extracted_kps.json")
# current InstantID 'front' for reference: eyes (0.446,0.523)(0.576,0.517) -> iocular 0.130, eyeY 0.520
print("current InstantID front baseline: iocular 0.130, eyeY 0.520 (the 'too small / too low' problem)")
