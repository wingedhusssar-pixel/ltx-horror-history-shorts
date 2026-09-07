import platform as _platform
_platform._wmi = None
_platform.uname()

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import json
import sys

IMAGE_PATH = "tv_frame.png"
OUTPUT_PATH = "screen_mask_points.json"

class MaskTracer:
    def __init__(self, root, image_path):
        self.root = root
        self.image_path = image_path
        self.points = []

        self.pil_img = Image.open(image_path).convert("RGB")
        self.orig_w, self.orig_h = self.pil_img.size

        screen_h = root.winfo_screenheight() - 140
        self.scale = min(1.0, screen_h / self.orig_h)
        disp_w = int(self.orig_w * self.scale)
        disp_h = int(self.orig_h * self.scale)

        self.disp_img = self.pil_img.resize((disp_w, disp_h))
        self.tk_img = ImageTk.PhotoImage(self.disp_img)

        self.canvas = tk.Canvas(root, width=disp_w, height=disp_h, cursor="cross")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        self.label = tk.Label(
            root,
            text="Click points tracing the screen's visible edge, following the curve. "
                 "Click close together on curved sections, fewer points on straight sections. "
                 "Press Enter when done (needs 8+ points). Press u to undo, q to quit.",
            font=("Arial", 11), wraplength=disp_w
        )
        self.label.pack(pady=6)

        self.count_label = tk.Label(root, text="Points: 0", font=("Arial", 12, "bold"))
        self.count_label.pack()

        self.canvas.bind("<Button-1>", self.on_click)
        root.bind("<Key-u>", self.undo)
        root.bind("<Key-q>", lambda e: root.destroy())
        root.bind("<Return>", self.finish)

    def on_click(self, event):
        orig_x = round(event.x / self.scale)
        orig_y = round(event.y / self.scale)
        self.points.append((orig_x, orig_y))
        self.redraw()

    def undo(self, event):
        if self.points:
            self.points.pop()
            self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        for i, (x, y) in enumerate(self.points):
            dx, dy = x * self.scale, y * self.scale
            r = 3
            self.canvas.create_oval(dx - r, dy - r, dx + r, dy + r, outline="red", fill="red")
        if len(self.points) >= 2:
            for i in range(len(self.points)):
                j = (i + 1) % len(self.points)
                x1, y1 = self.points[i][0] * self.scale, self.points[i][1] * self.scale
                x2, y2 = self.points[j][0] * self.scale, self.points[j][1] * self.scale
                self.canvas.create_line(x1, y1, x2, y2, fill="yellow", width=1)
        self.count_label.config(text=f"Points: {len(self.points)}")

    def finish(self, event):
        if len(self.points) < 8:
            self.count_label.config(text=f"Points: {len(self.points)} (need at least 8)")
            return
        self.save()
        self.root.destroy()

    def save(self):
        data = {
            "image_path": self.image_path,
            "image_size": [self.orig_w, self.orig_h],
            "polygon": [list(p) for p in self.points],
        }
        with open(OUTPUT_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.points)} polygon points to {OUTPUT_PATH}")


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    root = tk.Tk()
    root.title("Trace the screen edge")
    MaskTracer(root, image_path)
    root.mainloop()


if __name__ == "__main__":
    main()
