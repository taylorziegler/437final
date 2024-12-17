import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

def empty_dir(dir_path):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

class HoughDetector():
    def __init__(self) -> None:
        super().__init__()
        self.canny_lthreshold = 40
        self.canny_hthreshold = 90
        self.image_size = (1640, 1232)
        self.mask_level = 620

        self.image_name = None
        self.origin_image = None
        self.edges_image = None
        self.masked_edges_image = None
        self.output_image = None
        self.points = None
        
        self.out_dir = ""
        
        
    def _detect_edge(self):
        # Canny edge detection
        gray = cv2.cvtColor(self.origin_image, cv2.COLOR_BGR2GRAY)
        # # save the gray image
        # cv2.imwrite(os.path.join(self.out_dir, "gray.jpg"), gray)

        self.edges_image = cv2.Canny(gray, self.canny_lthreshold, self.canny_hthreshold)

        # # save the edges image
        # cv2.imwrite(os.path.join(self.out_dir, "edges.jpg"), self.edges_image)

    def _mask(self, vertics):
        # Make a square mask to make the detector focus on the road area
        mask = np.zeros_like(self.edges_image)
        mask_color = 255
        cv2.fillPoly(mask, [vertics], mask_color)
        return mask
    
    def _get_masked_image(self):
        # Filter the image with a square mask
        left_bottom = [0, self.image_size[1]]
        right_bottom = [self.image_size[0], self.image_size[1]]
        left_top = [0, self.mask_level]
        right_top = [self.image_size[0], self.mask_level]
        # apex = [self.image_size[1]/2, 500]
        vertices = np.array([left_bottom, right_bottom, right_top, left_top], np.int32)
        mask = self._mask(vertices)
        self.masked_edges_image = cv2.bitwise_and(self.edges_image, mask)

        # # save the masked image
        # cv2.imwrite(os.path.join(self.out_dir, "masked_edges.jpg"), self.masked_edges_image)
        
        
    def _get_points(self):
        # Do hough detection on the masked edges
        # lines = cv2.HoughLines(self.masked_edges_image, self.rho, self.theta, self.threshold, self.min_line_length, self.max_line_gap)
        lines = cv2.HoughLinesP(self.masked_edges_image, 1, np.pi / 180, threshold=50, minLineLength=150, maxLineGap=200)
        # print(lines)
        final_midpoint = None
        if lines is not None and len(lines) != 0:
            # draw lines using the start and end points of the lines
            self.output_image = np.copy(self.origin_image)  # copy the original image to draw lines on it
            lines = lines[:, 0]  

            slopes = np.array([self.calculate_slope(line) for line in lines]).reshape(-1)
            lines = np.array(lines, dtype=np.int32)
            # find the vertical lines and horizontal lines
            vertical_lines = lines[np.where(abs(slopes)>0.1)]  
            vertical_slopes = slopes[np.where(abs(slopes)> 0.1) ]
            horizontal_lines = lines[np.where(abs(slopes)<= 0.1) ]

            # if horizontal lines detected, calculate the midpoint as the target point
            if len(horizontal_lines) >= 1:
                final_midpoint = self.calculate_midpoint(horizontal_lines)
                cv2.circle(self.output_image, tuple(final_midpoint), 7, (0, 0, 255), -1)  
                for [x1, y1, x2, y2] in horizontal_lines:
                    cv2.line(self.output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                return final_midpoint
        
            # use k-means to cluster the vertical lines into two groups
            kmeans = KMeans(n_clusters=2, random_state=0).fit(vertical_slopes.reshape(-1,1))
            labels = kmeans.labels_

            group1 = vertical_lines[labels == 0]
            group2 = vertical_lines[labels == 1]
            
            # calculate the midpoint of two groups
            midpoint1 = self.calculate_midpoint(group1)
            midpoint2 = self.calculate_midpoint(group2)

            final_midpoint = ((midpoint1[0] + midpoint2[0]) // 2, (midpoint1[1] + midpoint2[1]) // 2)
            final_midpoint = np.array(final_midpoint)

            # cv2.circle(self.output_image, tuple(midpoint1), 5, (0, 255, 0), -1)
            # cv2.circle(self.output_image, tuple(midpoint2), 5, (255, 0, 0), -1)
            cv2.circle(self.output_image, tuple(final_midpoint), 7, (0, 0, 255), -1)

            for [x1, y1, x2, y2] in vertical_lines:
                    cv2.line(self.output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return final_midpoint
                    
    def save_images(self):
        # To see if it works
        cv2.imwrite(os.path.join(self.out_dir, "edges", self.image_name), self.edges_image)
        cv2.imwrite(os.path.join(self.out_dir, "mask", self.image_name), self.masked_edges_image)
        cv2.imwrite(os.path.join(self.out_dir, "out", self.image_name), self.output_image)

    def calculate_slope(self, line):
        """
        line: (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:  # 避免垂直直线的除零问题
            return float('inf')
        return (y2 - y1) / (x2 - x1)
    
    def calculate_midpoint(self, group):
        midpoints = []
        for x1, y1, x2, y2 in group:
            midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            midpoints.append(midpoint)
        return np.mean(midpoints, axis=0).astype(int)
    
    def process_image(self, image):
        self.origin_image = image
        self._detect_edge()
        self._get_masked_image()
        self.points = self._get_points()
        return self.output_image
    
    def projection_2_ground(self):
        # project the pixel coordinate into the ground plane
        H = np.array([[-2.48513348e+03, -1.44153041e+03,  1.91308005e+01], 
                      [-9.25354877e+01, -1.10178418e+03, -9.57331535e+01], 
                      [-9.16116864e-02, -1.90221209e+00,  1.34822475e-02]])
        
        u, v = self.points

        A1 = H[0, 0] - u * H[2, 0]
        B1 = H[0, 1] - u * H[2, 1]
        C1 = -(H[0, 2] - u * H[2, 2])

        A2 = H[1, 0] - v * H[2, 0]
        B2 = H[1, 1] - v * H[2, 1]
        C2 = -(H[1, 2] - v * H[2, 2])

        M = np.array([[A1, B1], [A2, B2]])
        C = np.array([C1, C2])

        X, Y = np.linalg.solve(M, C)

        return (X, Y)
    
# def main():
#     input_dir_path = "./input"
#     my_detector = HoughDetector()
#     for img_path in sorted(os.listdir(input_dir_path)):
#         if ".png" in img_path:
#             print("------", img_path, "------")
#             image = cv2.imread(os.path.join(input_dir_path, img_path))
#             my_detector.process_image(image, img_path)
#             my_detector.save_images()
    
# if __name__ == "__main__":
#     main()
