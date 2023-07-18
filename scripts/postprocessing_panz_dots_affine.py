from modules import scripts_postprocessing, shared
import gradio as gr
from PIL import Image
import cv2
import numpy as np
import os

class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
    name = "PanzDotsAffine"
    order = 20001
    # model = None

    def ui(self):

        with gr.Accordion('Panz Dots Affine', open=False):
            enable_dots_affine = gr.Checkbox(label="Enable dots affine", value=False)
            dots_input_dir = gr.Textbox(label="Dots Input directory", **shared.hide_dirs, placeholder="A group of dot images for estimating the affine transformation matrix.", elem_id="dots_input_dir")
            warp_target_dir = gr.Textbox(label="Warp Target directory", **shared.hide_dirs, placeholder="A group of target images to be affine transformed.", elem_id="warp_target_dir")
            output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="output folder.", elem_id="output_dir")

        return {
            "enable_dots_affine": enable_dots_affine,
            "dots_input_dir": dots_input_dir,
            "warp_target_dir": warp_target_dir,
            "output_dir": output_dir,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, enable_dots_affine, dots_input_dir, warp_target_dir, output_dir):
        # if not model or model == "None":
        #     return

        if enable_dots_affine:

            cXr = 0.0
            cYr = 0.0
            cXg = 0.0
            cYg = 0.0
            cXb = 0.0
            cYb = 0.0

            # Reference Dots Image

            # image split
            w, h = pp.image.size
            image = np.array(pp.image)
            r1, g1, b1 = image[:, :, 0], image[:, :, 1], image[:, :, 2]

            r = np.zeros((w * 2, h * 2))
            g = np.zeros((w * 2, h * 2))
            b = np.zeros((w * 2, h * 2))

            half_w = int(w/2)
            half_h = int(h/2)

            r[half_w:half_w+w, half_h:half_h+h] = r1
            g[half_w:half_w+w, half_h:half_h+h] = g1
            b[half_w:half_w+w, half_h:half_h+h] = b1

            ret, r = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY)
            ret, g = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
            ret, b = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)

            # Calculate R moments
            mr = cv2.moments(r)
            if not mr["m00"] == 0:
                # Calculate x,y coordinates of centre
                cXr = int(mr["m10"] / mr["m00"])
                cYr = int(mr["m01"] / mr["m00"])
                cv2.circle(image,
                           center=(cXr, cYr),
                           radius=20,
                           color=(255, 0, 0),
                           thickness=2,
                           lineType=cv2.LINE_4,
                           shift=0)
                print(f'R Centroid at location: {cXr},{cYr}')

            # Calculate G moments
            mg = cv2.moments(g)
            if not mg["m00"] == 0:
                # Calculate x,y coordinates of centre
                cXg = int(mg["m10"] / mg["m00"])
                cYg = int(mg["m01"] / mg["m00"])
                cv2.circle(image,
                           center=(cXg, cYg),
                           radius=20,
                           color=(0, 255, 0),
                           thickness=2,
                           lineType=cv2.LINE_4,
                           shift=0)
                print(f'G Centroid at location: {cXg},{cYg}')

            # Calculate B moments
            mb = cv2.moments(b)
            if not mb["m00"] == 0:
                # Calculate x,y coordinates of centre
                cXb = int(mb["m10"] / mb["m00"])
                cYb = int(mb["m01"] / mb["m00"])
                cv2.circle(image,
                           center=(cXb, cYb),
                           radius=20,
                           color=(0, 0, 255),
                           thickness=2,
                           lineType=cv2.LINE_4,
                           shift=0)
                print(f'B Centroid at location: {cXb},{cYb}')

            src = np.array([[cXr, cYr], [cXg, cYg], [cXb, cYb]], np.float32)

            image_list = shared.listfiles(dots_input_dir)
            image_list2 = shared.listfiles(warp_target_dir)

            counter = 0

            for filename in image_list:

                pil_warp_target = Image.open(image_list2[counter]) #.convert('BGR')
                warp_target1 = np.array(pil_warp_target)

                warp_target = 255 * np.ones((w * 2, h * 2, 4))
                # warp_target[half_w:half_w+w, half_h:half_h+h, :] = warp_target1
                warp_target[half_w:half_w+w, half_h:half_h+h, 0] = warp_target1[:, :, 2]
                warp_target[half_w:half_w+w, half_h:half_h+h, 1] = warp_target1[:, :, 1]
                warp_target[half_w:half_w+w, half_h:half_h+h, 2] = warp_target1[:, :, 0]

                try:
                    pil_image = Image.open(filename) #.convert('RGB')
                except Exception:
                    continue

                # r, g, b = image.split()
                image = np.array(pil_image)
                r1, g1, b1 = image[:, :, 0], image[:, :, 1], image[:, :, 2]

                r = np.zeros((w * 2, h * 2))
                g = np.zeros((w * 2, h * 2))
                b = np.zeros((w * 2, h * 2))

                r[half_w:half_w+w, half_h:half_h+h] = r1
                g[half_w:half_w+w, half_h:half_h+h] = g1
                b[half_w:half_w+w, half_h:half_h+h] = b1

                ret, r = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY)
                ret, g = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
                ret, b = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)

                # Calculate R moments
                mr = cv2.moments(r)

                if not mr["m00"] == 0:

                    # Calculate x,y coordinates of centre
                    cXr = int(mr["m10"] / mr["m00"])
                    cYr = int(mr["m01"] / mr["m00"])

                    cv2.circle(image,
                               center=(cXr, cYr),
                               radius=20,
                               color=(255, 0, 0),
                               thickness=2,
                               lineType=cv2.LINE_4,
                               shift=0)

                    #print(f'R Centroid at location: {cXr},{cYr}')

                # Calculate G moments
                mg = cv2.moments(g)

                if not mg["m00"] == 0:

                    # Calculate x,y coordinates of centre
                    cXg = int(mg["m10"] / mg["m00"])
                    cYg = int(mg["m01"] / mg["m00"])

                    cv2.circle(image,
                               center=(cXg, cYg),
                               radius=20,
                               color=(0, 255, 0),
                               thickness=2,
                               lineType=cv2.LINE_4,
                               shift=0)

                    #print(f'G Centroid at location: {cXg},{cYg}')

                # Calculate B moments
                mb = cv2.moments(b)

                if not mb["m00"] == 0:

                    # Calculate x,y coordinates of centre
                    cXb = int(mb["m10"] / mb["m00"])
                    cYb = int(mb["m01"] / mb["m00"])

                    cv2.circle(image,
                               center=(cXb, cYb),
                               radius=20,
                               color=(0, 0, 255),
                               thickness=2,
                               lineType=cv2.LINE_4,
                               shift=0)

                    #print(f'B Centroid at location: {cXb},{cYb}')

                print(f'Centroid at location: R:({cXr},{cYr}) G:({cXg},{cYg}) B:({cXb},{cYb})')

                # ImShow For Debug
                # cv2.imshow("Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #image)
                # cv2.waitKey(0)

                dst = np.array([[cXr, cYr], [cXg, cYg], [cXb, cYb]], np.float32)

                affine = cv2.getAffineTransform(dst, src) #src, dst)

                warp_target = cv2.warpAffine(warp_target, affine, dsize=(2048, 2048), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE, borderValue=(255, 255, 255))

                # Save output file
                # cv2.imwrite('D:\\result' + str(counter) + '.png', warp_target)

                # Check whether the specified path exists or not
                isExist = os.path.exists(output_dir)
                if not isExist:                
                    # Create a new directory because it does not exist
                    os.makedirs(output_dir)
                    print("The new directory is created!")
                cv2.imwrite(output_dir + "\\" + str(counter+11).zfill(5) + '.png', warp_target)

                counter += 1

            print("Panz Dots Affine finished!")

        return
