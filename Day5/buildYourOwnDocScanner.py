#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:39:11 2026

@author: akshitmudgal
"""

"""

> **A Quick Note Before You Revisit This Project**
>
> When you come back to this, there may be moments where you feel stuck and wonder:
>
> *"Why are we using this function?"*
>
> *"What problem was this step solving?"*
>
> *"How did we arrive at this calculation?"*
>
> That's completely normal. In fact, it happens to almost everyone when learning Image Processing, OpenCV, or any new technical subject.
>
> Don't try to memorize the entire code. Instead, focus on understanding the flow:
>
> ```text
> What problem do we have?
>      
> Which function solves it?
>      
> How does that function work internally?
>      
> What output does it produce?
>      
> Why is that output needed in the next step?
> ```
>
> If you feel stuck, try to answer:
>
> 1. What is the input?
> 2. What is the output?
> 3. Why do we need it?
> 4. What would happen if we removed it?
>
> These four questions alone can help you understand most of the pipeline.
>
> Learning is not about remembering every line of code. It is about understanding the reasoning behind the code. Once the reasoning becomes clear, the code becomes much easier to recall.
>
> So if something doesn't make sense immediately, don't get frustrated. Spend some time exploring it, experiment with different values, observe the outputs, and try to connect the dots. That process is where the real learning happens.
>
> And if a step still feels confusing after your own investigation, make a note of it and bring it up in the next session. 
Questions are not a sign that you're behind—they're often a sign that you're thinking deeply about how the system actually works. 



"""

#DOC SCANNER

import cv2
import numpy as np
import matplotlib.pyplot as plt

#STEP 1: TO LOAD THE IMAGE

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not open the image:{path}")
    return image

def resize_image(image, height = 700):
    ratio = image.shape[0] / height
    width = int(image.shape[1] * (height/ image.shape[0]))
    resized = cv2.resize(image, (width, height))
    return resized, ratio
    
def preprocess(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations = 1)
    return edges

def find_document_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    
    for contour in contours[:10]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            return approx
    return None;

#isey fir se smjh lenge no worries :)
def order_corner(pts):
    pts = pts.reshape(4,2).astype(np.float32)
    ordered = np.zeros((4,2), dtype = np.float32)
    
    #bottom right and to left 
    s = pts.sum(axis= 1)
    ordered[0] = pts[np.argmin(s)] #tl
    ordered[2] = pts[np.argmax(s)] #br
    
    diff = np.diff(pts, axis= 1)
    ordered[1] = pts[np.argmin(diff)] #tr
    ordered[3] = pts[np.argmax(diff)] #bl
    
    return ordered #TL, TR, BR, BL
    

def warp_document(image, corners):
        tl, tr, br, bl = corners
        
        W = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
        H = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
        
        dst = np.array([
            [0,0], 
            [W-1,0],
            [W-1, H-1],
            [0,H-1]
            
            ],dtype = np.float32)
        
        #perspective matrix
        M = cv2.getPerspectiveTransform(corners, dst)
        return cv2.warpPerspective(image, M, (W,H))

def make_scan_look(warped):
    
    #CLAHE -> FIXING THE UNEVEN LIGHTING
    #DENOISE ->
    #OTSU -> AUTOMATICALLY FIND THE BEST THRESHOLD TO MAKE IT BLACK AND WHITE
    
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    #clahe
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    enhanced = clahe.apply(gray)
    
    #DeNoise Fst Non LOcal denoising
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    #theshold
    _, scanned = cv2.threshold(enhanced, 0, 255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    
    return scanned

def show_results(original, edges, detected, warped, scan):
    orig_rgb = cv2.cvtColor(original,cv2.COLOR_BGR2RGB )
    detected_rgb = cv2.cvtColor(detected,cv2.COLOR_BGR2RGB )
    warped_rgb = cv2.cvtColor(warped,cv2.COLOR_BGR2RGB )
    
    fig, axes = plt.subplots(1, 5, figsize=(20,5))
    fig.suptitle("Doc scanner - IGDTUW", fontsize = 14, fontweight = "bold")
    
    steps = [
        (orig_rgb, "Step1: Original"),
        (edges, "Step1: Edges"),
        (detected_rgb, "Step1: detectd rgb"),
        (warped_rgb, "Step1: warped"),
        (scan, "Step5: final scan"),
        ]
    for ax, (img, title) in zip(axes, steps):
        ax.imshow(img, cmap="gray" if len(img.shape)==2 else None)
        ax.set_title(title, fontsize = 10)
        ax.axis("off")
        
    plt.tight_layout()
    plt.savefig("scan_steps.png", dpi = 150, bbox_inches ="tight")
    plt.show()
    print("Done done saved as scan_steps.png")
    
    
    

def scan_document(image_path):
    print(f"\n Sacnning: {image_path}")
    print("--" * 40)
    
    #load
    print("1. Loading the image....")
    original = load_image(image_path)
    print(f"  Size: {original.shape[1]} * {original.shape[0]} px")
    
    #Resize for dtection
    print("2. Resizing for detection")
    small, ratio = resize_image(original, height = 700)
    
    #pre-process
    print("3. kuch preprocessing jaise: edge detection ")
    edges = preprocess(small)
    
    #find the outline
    print("4. Finding doc outline")
    contour = find_document_contour(edges)
    
    if contour is None:
        print(" Could not find a outline of doc")
        print(" Tips: ")
        print(" Place doc on a dark bg")
        print(" Make sure all 4 corners are visible")
        print("Improve lighting, avoid shadows")
    
    print(" Found the 4 cornered outline!")
    
    detected = small.copy()
    cv2.drawContours(detected, [contour], -1, (0,255,0), 3)
    for point in contour.reshape(4,2):
        cv2.circle(detected, tuple(point.astype(int)), 8,(0,0,255), -1)
    
    print(". Applying perspective corrrection...")
    #scaling corners back to orginial resoluition
    corners = order_corner(contour * ratio)
    warped = warp_document(original, corners)
    
    #create a scan effect
    print(" 6. apply scan effect : CLAHE, Otsu thresholding")
    scan = make_scan_look(warped)
    
    cv2.imwrite("scanned-document.png", scan)
    print(" WE have saved your file")
    
    #show
    print("7. Shoiwng results")
    show_results(original, edges, detected, warped, scan)
    
    
if __name__ == "__main__":
    IMAGE_PATH = "images/input/hwdoc.jpg"
    scan_document(IMAGE_PATH)