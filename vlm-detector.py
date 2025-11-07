# vlm_detector.py
import base64, json, requests, cv2
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Detection:
    label: str
    score: float
    bbox: Tuple[int,int,int,int]  # (u1,v1,u2,v2)

class VLMDetector:
    def __init__(self, url="http://saltyfish.eecs.umich.edu:8000/v1/chat/completions",
                 model="Qwen/Qwen3-VL-30B-A3B-Instruct", timeout=20):
        self.url = url
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _b64_from_bgr(bgr, jpeg_quality=85):
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return base64.b64encode(enc.tobytes()).decode("utf-8")

    def _build_prompt(self, W, H, categories: List[str]):
        cats = ", ".join(sorted(set(c.lower() for c in categories)))
        return (
            "You are an object detector. Given ONE or MORE images of the same scene (RGB and optionally a depth colormap), "
            f"detect ONLY these categories: [{cats}].\n"
            "Return STRICT JSON as: {\"detections\":[{\"label\":\"<one of categories>\","
            "\"score\":0.0-1.0,\"bbox\":[u1,v1,u2,v2]}...]}\n"
            f"Coordinates are INTEGER PIXELS in the ORIGINAL RGB size (width={W}, height={H}). "
            "Ensure 0<=u1<u2<width and 0<=v1<v2<height. If nothing is present, return {\"detections\":[]}."
        )

    def detect(self, bgr_rgb, categories: List[str], extra_images_bgr: List = None,
               temperature=0.0, top_p=1.0):
        H, W = bgr_rgb.shape[:2]
        prompt = self._build_prompt(W, H, categories)

        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._b64_from_bgr(bgr_rgb)}"}}]

        if extra_images_bgr:
            for img in extra_images_bgr:
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{self._b64_from_bgr(img)}"}})

        data = {
            "model": self.model,
            "messages": [{"role":"user","content": content}],
            "temperature": temperature,
            "top_p": top_p
        }
        r = requests.post(self.url, json=data, timeout=self.timeout)
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"].strip()

        if reply.startswith("```"):
            lines = reply.splitlines()
            if lines and lines[-1].strip().startswith("```"):
                reply = "\n".join(lines[1:-1]).strip()
            else:
                reply = "\n".join(lines[1:]).strip()

        obj = json.loads(reply)
        outs: List[Detection] = []
        for d in obj.get("detections", []):
            lab = str(d["label"]).lower().strip()
            u1,v1,u2,v2 = [int(round(x)) for x in d["bbox"]]
            u1 = max(0, min(u1, W-1)); u2 = max(0, min(u2, W-1))
            v1 = max(0, min(v1, H-1)); v2 = max(0, min(v2, H-1))
            if u2 <= u1 or v2 <= v1:
                continue
            outs.append(Detection(lab, float(d.get("score", 0.0)), (u1,v1,u2,v2)))
        return outs
