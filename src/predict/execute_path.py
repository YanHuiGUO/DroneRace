

if __name__== '__main__':   
    # Define the input limits:
    fmin = 0.1  #[m/s**2]
    fmax = 2 #[m/s**2]
    wmax = 0.79 #[rad/s]
    minTimeSec = 0.02 #[s]

    # Define how gravity lies:
    gravity = [0,0,-9.81]
    path_handle = Generate_Path(fmin,fmax,wmax, minTimeSec,gravity)
    con = Commander()
    img = Image_Capture()
    jump_once = 1
    theta = 0
    r = 2
    
    c_x,c_y = 10,10
    bias_x,bias_y = -0.1,0.5
    start_x = c_x -r +bias_x
    start_y = c_y + bias_y
    sin_theta = np.sin(np.deg2rad(theta))
    cos_theta = np.cos(np.deg2rad(theta))
    print (sin_theta,cos_theta)
    next_x = r-r * cos_theta + start_x
    next_y = -r * sin_theta + start_y
   
    mav = MAV_Jump_Ring(next_x,next_y,2,theta)
    '''
    init the mav position 
    '''
    for i in range(10):
        con.move(mav.init_x,mav.init_y,mav.init_z,mav.init_yaw,False)
        time.sleep(0.02)
    time.sleep(10)


    
    while 1:
        mav.run(con, img,path_handle)
