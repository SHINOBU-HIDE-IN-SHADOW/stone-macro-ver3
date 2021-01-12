import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageGrab
import pyautogui
import time
import keyboard
run = True
while run:
        #pyautogui fail safe
        pyautogui.PAUSE = 0.1
        #end program
        if keyboard.is_pressed("k"): 
                break
        wind = np.array(ImageGrab.grab(bbox=(0,0,1365,767)))
        wind = cv2.cvtColor(wind, cv2.COLOR_RGB2GRAY)
        hero_power = cv2.imread('D:\\Users\\lobot\Desktop\\pyprojects\\macro test\\shield.JPG', 0)
        game_start = cv2.imread('D:\\Users\\lobot\\Desktop\\pyprojects\\macro test\\start.JPG', 0)
        card_warrior = cv2.imread('D:\\Users\\lobot\\Desktop\\pyprojects\\macro test\\minionw.PNG', 0)
        card_all = cv2.imread('D:\\Users\\lobot\\Desktop\\pyprojects\\macro test\\minionwil.PNG', 0)
        hero = cv2.imread('D:\\Users\\lobot\\Desktop\\pyprojects\\macro test\\warriiorh.JPG', 0)

        #hero power
        sift = cv2.xfeatures2d.SIFT_create()        
        kp_hpower1, des_hpower1 = sift.detectAndCompute(hero_power,None)
        kp_hpower2, des_hpower2 = sift.detectAndCompute(wind,None)        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_hpower1,des_hpower2, k=2)
        match_pts_hpower = []
        for m1, m2 in matches:
                if m1.distance < 0.3*m2.distance:
                        idx = m1.trainIdx
                        match_pts_hpower.append(kp_hpower2[idx].pt)
                
        if len(match_pts_hpower) != 0:
                match_pts_hpower = np.array(match_pts_hpower)
                pyautogui.click(match_pts_hpower[0, 0], match_pts_hpower[0, 1], button = "left")
                

        #gmae start
        sift2 = cv2.xfeatures2d.SIFT_create()        
        kp_gstart1, des_gstart1 = sift2.detectAndCompute(game_start,None)
        kp_gstart2, des_gstart2 = sift2.detectAndCompute(wind,None)        
        bf2= cv2.BFMatcher()
        matches2 = bf2.knnMatch(des_gstart1,des_gstart2, k=2)
        match_pts_gstart = []
        for m1, m2 in matches2:
                if m1.distance < 0.3*m2.distance:
                        idx = m1.trainIdx
                        match_pts_gstart.append(kp_gstart2[idx].pt)
                
        if len(match_pts_gstart) != 0:
                match_pts_gstart = np.array(match_pts_gstart)
                pyautogui.click(match_pts_gstart[0, 0], match_pts_gstart[0, 1], button = "left")
                

        #card player
        sift3 = cv2.xfeatures2d.SIFT_create()        
        kp_wcard1, des_wcard1 = sift3.detectAndCompute(card_warrior,None)
        kp_wcard2, des_wcard2 = sift3.detectAndCompute(wind,None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches3 = flann.knnMatch(des_wcard1,des_wcard2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = []
        # ratio test as per Lowe's paper
        for m1,m2 in matches3:
                if m1.distance < 0.633*m2.distance:
                        idx = m1.trainIdx
                        matchesMask.append(kp_wcard2[idx].pt)
                
        if len(matchesMask) != 0:
                matchesMask = np.array(matchesMask)
                pyautogui.click(matchesMask[0, 0],matchesMask[0, 1], button = "left") 
                pyautogui.dragTo(matchesMask[0, 0]-200,matchesMask[0, 1]-200, button = "left") 

        #play card all
        sift4 = cv2.xfeatures2d.SIFT_create()        
        kp_cardall1, des_cardall1 = sift4.detectAndCompute(card_all,None)
        kp_cardall2, des_cardall2 = sift4.detectAndCompute(wind,None)        
        FLANN_INDEX_KDTREE = 0
        index_params2 = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params2 = dict(checks=50)   # or pass empty dictionary
        flann2 = cv2.FlannBasedMatcher(index_params2,search_params2)
        matches4 = flann2.knnMatch(des_cardall1,des_cardall2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask2 = []
        # ratio test as per Lowe's paper
        for m1,m2 in matches3:
                if m1.distance < 0.633*m2.distance:
                        idx = m1.trainIdx
                        matchesMask2.append(kp_wcard2[idx].pt)
                
        if len(matchesMask2) != 0:
                matchesMask2 = np.array(matchesMask2)
                pyautogui.click(matchesMask2[0, 0],matchesMask2[0, 1], button = "left") 
                pyautogui.dragTo(matchesMask2[0, 0]-200,matchesMask2[0, 1]-200, button = "left")  
        
        #hero click
        sift5 = cv2.xfeatures2d.SIFT_create()        
        kp_hero1, des_hero1 = sift5.detectAndCompute(hero,None)
        kp_hero2, des_hero2 = sift5.detectAndCompute(wind,None)        
        bf3 = cv2.BFMatcher()
        matches5 = bf3.knnMatch(des_hero1,des_hero2, k=2)
        match_pts_hero = []
        for m1, m2 in matches5:
                if m1.distance < 0.4*m2.distance:
                        idx = m1.trainIdx
                        match_pts_hero.append(kp_hero2[idx].pt)
                
        if len(match_pts_hero) != 0:
                match_pts_hero = np.array(match_pts_hero)
                pyautogui.click(match_pts_hero[0, 0], match_pts_hero[0, 1], button = "left") 
        


cv2.destroyAllWindows()
