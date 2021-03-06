from utils import *
if __name__ == "__main__":
    #Set configurations and load models####################################
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR\tesseract'
    model1 = load_model('models/digit_deta.h5') 
    model2 = load_model('models/mnist_model_toxo.h5') 
    
    type_s = "Sudokus/digital_sudokus/" #handwritten  #digital_sudokus
    name = "Captura (2)"
    #######################################################################
    
    #read the image########################################################
    img = cv2.imread('figs/' + type_s + name + '.png', 1)
    #img = cv2.resize(img, (495, 495))
    imgC = img.copy()
    imgOriginal = img.copy()
    plt.imshow(cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("Img Input")
    if not os.path.exists('results/' + name):
        os.mkdir('results/' + name)
    plt.savefig('results/' + name + '/'+ '1_input_img.png', dpi=300, bbox_inches='tight'),plt.show()
    #######################################################################
    
    #warp the image########################################################
    imgVoid = np.zeros((495, 495, 3), np.uint8) #void image to warp the original to
    gray = basic_binarize(img) #process the input image a bit to find contours better
    
    #plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)), plt.show()
    
    imgC = img.copy() #copy to plot
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find all contours
    cv2.drawContours(imgC, contours, -1, (0, 255, 0), 3) #draw all contours in green


    SudokuPoints = SudokuPoints(contours) #get all the points forming contours and return the biggest one in order
    
    #print(SudokuPoints) #print the points
    
    cv2.drawContours(imgC, SudokuPoints, -1, (0, 0, 255), 25) #draw in red the 4 points that form the sudoku
    plt.imshow(cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)), plt.axis('off'),plt.title("Img biggest contour")
    plt.savefig('results/' + name + '/'+ '2_Img_biggest_contour.png', dpi=300, bbox_inches='tight'),plt.show()

    pointsSudoku = np.float32(SudokuPoints) 
    pointsVoid = np.float32([[0, 0],[495, 0], [0, 495],[495, 495]]) 
    matrix = cv2.getPerspectiveTransform(pointsSudoku, pointsVoid)
    imgWarpColor = cv2.warpPerspective(img, matrix, (495, 495))
    plt.imshow(cv2.cvtColor(imgWarpColor, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("Img warped")
    plt.savefig('results/' + name + '/'+ '3_Img_warped.png', dpi=300, bbox_inches='tight'),plt.show()

    img = cv2.cvtColor(imgWarpColor,cv2.COLOR_BGR2GRAY)
    #######################################################################

    
    #binarize img##########################################################
    img_binary = get_binary_img(img)
    
    plt.imshow(cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("Img binarized")
    plt.savefig('results/' + name + '/'+'4_binarized_img.png', dpi=300, bbox_inches='tight'),plt.show()
    #######################################################################
    
    '''
    # Detect the contours in the image#####################################
    contours, _ = cv2.findContours(img_binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    
    # Iterate through all the contours
    for contour in contours:
        # Find bounding rectangles
        x,y,w,h = cv2.boundingRect(contour)
        # Draw the rectangle
        cv2.rectangle(img_binary,(x,y),(x+w,y+h),100,1)
    
    plt.imshow(cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("Img binarized2"),plt.show()
    #######################################################################
    '''
    # Procces each box (sudoku cell) and plot them (image processing)######
    img_c = img_binary.copy() 
    boxes = splitBoxes(img_c)
    boxes_np, boxes = image_proc_boxes(boxes)
    plot_boxes(boxes_np, name)
    #######################################################################
    
    #Predict DIGITS with TensorFlow AND TESSERACT##########################
    boxes_prob, numbers, mean_confidence_models = Predict(model1, model2, boxes_np, boxes, custom_config)
    plot_prob(boxes_prob, numbers, name)
    sudoku = np.asarray(numbers)
    sudoku_m = sudoku.copy()
    sudoku_m = np.reshape(sudoku_m, (9,9))
    #######################################################################
    
    #GET FINAL ACCURACY and MEAN CONFIDENCE ###############################
    
    print('The Mean confidence obtained with the different models when predicting the digits in the sudoku is:')
    print('Model1 - Digit_deta - Computer Digits:', mean_confidence_models[0])
    print('Model2 - Mnist - Handwritten Digits:', mean_confidence_models[1])
    print('Tesseract OCR', mean_confidence_models[2])

    print('Accuracy obtained (Numbers well predicted / numbers to predict):', get_accuracy(name, type_s, sudoku_m))
    
    #######################################################################
    
    
    #Solve Sudoku##########################################################
    
    IsThereSolution = Solve(0, sudoku);
    if IsThereSolution == False:
        print("no solution to this sudoku")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("Img result")
        plt.savefig('results/' +name + '/'+ '7_img_result.png', dpi=300, bbox_inches='tight'),plt.show()
        exit()
    else:
        sudoku_m_solved = sudoku.copy()
        sudoku_m_solved = np.reshape(sudoku, (9,9))
        sudoku_solution = sudoku_m_solved - sudoku_m
    #######################################################################
    
    #PUT SOLUTION TO THE ORIGINAL IMAGE####################################
    img_c2 = img.copy() 
    img_c = np.zeros((img_c2.shape[0],img_c2.shape[1],3))
    img_c[:,:,0] = img_c2
    img_c[:,:,1] = img_c2
    img_c[:,:,2] = img_c2
    imgVoid = np.zeros((495, 495, 3), np.uint8) #void image to warp the original to

    img_c = overlay_solution(imgVoid, sudoku_solution)
    img_c = img_c.astype('uint8')
    
    plt.imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("Img result")
    plt.savefig('results/' + name + '/'+'8_img_result.png', dpi=300, bbox_inches='tight'),plt.show()
            
    
    pointsSudoku = np.float32(SudokuPoints) 
    pointsImg = np.float32([[0, 0],[495, 0], [0, 495],[495, 495]]) 
    matrix = cv2.getPerspectiveTransform(pointsImg, pointsSudoku)
    imgOut = cv2.warpPerspective(img_c, matrix, (imgOriginal.shape[1], imgOriginal.shape[0]))
    
    '''
    imgVoidOut = cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY)
    imgVoidOut[imgVoidOut > 0] = 255
    '''
    imgaux = np.zeros((imgOut.shape[0],imgOut.shape[1],3), np.uint8)
    imgaux[:,:,0] = imgOut[:,:,1]
    imgaux[:,:,1] = imgOut[:,:,1]
    imgaux[:,:,2] = imgOut[:,:,1]
    
       
    imgedited = imgOriginal * cv2.bitwise_not(imgaux).astype('bool')
    imgedited[:,:,1] = (imgaux[:,:,1]/200).astype('uint8') + imgedited[:,:,1]
    
    
    plt.imshow(cv2.cvtColor(imgedited, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("Img output")
    plt.savefig('results/' + name + '/'+ '9_Img_output.png', dpi=300, bbox_inches='tight'),plt.show()

    
    #######################################################################
    
    
    
    







