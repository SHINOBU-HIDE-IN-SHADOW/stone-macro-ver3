import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageGrab
import pyautogui
import time
import keyboard
run = True
class macroclass:
    def __init__(self,images):
        sift = cv2.xfeatures2d.SIFT_create()        
        kp1, des1 = sift.detectAndCompute(images,None)
        kp2, des2 = sift.detectAndCompute(wind,None)        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        match_pts_screen = []
        for m1, m2 in matches:
                if m1.distance < 0.3*m2.distance:
                        idx = m1.trainIdx
                        match_pts_screen.append(kp2[idx].pt)
                
        if len(match_pts_screen) != 0:
                match_pts_screen = np.array(match_pts_screen)
                pyautogui.click(match_pts_screen[0, 0]+20, match_pts_screen[0, 1]+2, button = "left")
                time.sleep(0.7)
                pyautogui.click()
class cards:
    def __init__(self,cardimg):
        sift3 = cv2.xfeatures2d.SIFT_create()        
        kp_card1, des_card1 = sift3.detectAndCompute(cardimg,None)
        kp_card2, des_card2 = sift3.detectAndCompute(wind,None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 6)
        search_params = dict(checks=180)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches2 = flann.knnMatch(des_card1,des_card2,k=2)
        matchesMask = []
        for m1,m2 in matches2:
                if m1.distance < 0.7*m2.distance:
                        idx = m1.trainIdx
                        matchesMask.append(kp_card2[idx].pt)
                
        if len(matchesMask) != 0:
                matchesMask = np.array(matchesMask)
                pyautogui.click(matchesMask[0, 0],matchesMask[0, 1], button = "left") 
                pyautogui.dragTo(matchesMask[0, 0]-200,matchesMask[0, 1]-200, button = "left")
while run:
        #pyautogui fail safe
        pyautogui.PAUSE = 0.1
        #end program
        if keyboard.is_pressed("k"): 
                break
        screen_result = pyautogui.size()
        wind = np.array(ImageGrab.grab(bbox=(0,0,screen_result[0],screen_result[1])))
        wind = cv2.cvtColor(wind, cv2.COLOR_RGB2GRAY)
        hero_power = cv2.imread('D:\\Users\\lobot\Desktop\\pyprojects\\macro test\\shield.JPG', 0)
        game_start = cv2.imread('D:\\Users\\lobot\\Desktop\\pyprojects\\macro test\\start.JPG', 0)
        hero = cv2.imread('D:\\Users\\lobot\\Desktop\\pyprojects\\macro test\\warriiorh.JPG', 0)
        cost1 = cv2.imread('D:\\Users\\lobot\\Desktop\\pyprojects\\macro test\\cost1.PNG', 0)
        cost2 = cv2.imread('D:\\Users\\lobot\\Desktop\\pyprojects\\macro test\\cost2.PNG', 0)
        cost3 = cv2.imread('D:\\Users\\lobot\\Desktop\\pyprojects\\macro test\\cost3.PNG', 0)
        cost4 = cv2.imread('D:\\Users\\lobot\\Desktop\\pyprojects\\macro test\\cost4.PNG', 0)
        cost5 = cv2.imread('D:\\Users\\lobot\\Desktop\\pyprojects\\macro test\\cost5.PNG', 0)
        macroclass(hero_power)
        macroclass(game_start)
        macroclass(hero)
        cards(cost1)
        cards(cost2)
        cards(cost3)
        cards(cost4)
        cards(cost5)
            
cv2.destroyAllWindows()
