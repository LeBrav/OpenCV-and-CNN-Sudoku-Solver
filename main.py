from utils import *
if __name__ == "__main__":
    #Set configurations and load models####################################
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR\tesseract'
    model1 = load_model('models/digit_deta.h5') 
    model2 = load_model('models/mnist_model_toxo.h5') 
    name = "digital (1)"
    #######################################################################
    
    #read the image########################################################
    img = cv2.imread('figs/' + name + '.png', 0)
    img = cv2.resize(img, (495, 495))
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("Img Input")
    if not os.path.exists('results/' + name):
        os.mkdir('results/' + name)
    plt.savefig('results/' + name + '/'+ '1_input_img.png', dpi=300, bbox_inches='tight'),plt.show()
    #######################################################################
    
    #binarize img##########################################################
    img_binary = get_binary_img(img)
    
    plt.imshow(cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("Img binarized")
    plt.savefig('results/' + name + '/'+'2_binarized_img.png', dpi=300, bbox_inches='tight'),plt.show()
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
    boxes_prob, numbers = Predict(model1, model2, boxes_np, boxes, custom_config)
    plot_prob(boxes_prob, numbers, name)
    sudoku = np.asarray(numbers)
    #######################################################################
    
    
    #Solve Sudoku##########################################################
    sudoku_m = sudoku.copy()
    sudoku_m = np.reshape(sudoku_m, (9,9))
    
    IsThereSolution = Solve(0, sudoku);
    if IsThereSolution == False:
        print("no solution to this sudoku")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("Img result")
        plt.savefig('results/' +name + '/'+ '5_img_result.png', dpi=300, bbox_inches='tight'),plt.show()
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
    
    img_c = overlay_solution(img_c, sudoku_solution)
    img_c = img_c.astype('uint8')
    
    plt.imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title("Img result")
    plt.savefig('results/' + name + '/'+'5_img_result.png', dpi=300, bbox_inches='tight'),plt.show()
            
    
    #######################################################################
    
    







