import os, numpy as np, cv2
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="antelopev2", root=os.path.expanduser("~/.insightface"), providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640,640))
D="/tmp/instantid_exp"
for f in sorted(os.listdir(D)):
    if not f.endswith(".png") or "_kps" in f or "montage" in f: continue
    img=cv2.imread(os.path.join(D,f)); h,w=img.shape[:2]
    faces=app.get(img)
    if not faces: print(f"{f:<32} no face"); continue
    fc=max(faces,key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    k=fc.kps/np.array([w,h]); x1,y1,x2,y2=fc.bbox
    print(f"{f:<32} iocular {abs(k[1,0]-k[0,0]):.3f}  eyeY {k[:2,1].mean():.3f}  faceH {(y2-y1)/h*100:.0f}%")
