import openvino.inference_engine as ie 
import cv2
import numpy
import time
def main():
    #######################  Device  Initialization  ########################
    #  Plugin initialization for specified device and load extensions library if specified
    plugin = ie.IEPlugin(device="MYRIAD")
    #########################################################################

    #########################  Load Neural Network  #########################
    #  Read in Graph file (IR)
    net = ie.IENetwork(model="cnn-mnist_inference.xml", weights="cnn-mnist_inference.bin")

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    #  Load network to the plugin
    exec_net = plugin.load(network=net)
    del net
    ########################################################################

    #########################  Obtain Input Tensor  ########################
    #  Obtain and preprocess input tensor (image)
    #  Read and pre-process input image  maybe we don't need to show these details
    image_for_inference = cv2.imread("./1.JPG")
    
    image_for_inference = cv2.cvtColor(image_for_inference, cv2.COLOR_BGR2GRAY)
    image_for_inference=cv2.resize(image_for_inference, (28,28))

    image_for_inference = image_for_inference.astype(numpy.float32)

    image_for_inference[:] =1-((image_for_inference[:] )*(1.0/255.0))

    image_for_inference=image_for_inference.reshape(-1,28,28)
    # ########################################################################

    # ##########################  Start  Inference  ##########################
    # #  Start synchronous inference and get inference result
    start_time=time.time()
    req_handle = exec_net.start_async(0,inputs={input_blob:image_for_inference})
    # # ########################################################################
    # res = exec_net.infer({input_blob:image_for_inference})
    # # ######################## Get Inference Result  #########################
    status = req_handle.wait()
    res = req_handle.outputs[out_blob]


    # Do something with the results... (like print top 5)
    print("FPS:",1/(time.time()-start_time))
    
    print((1-res[0]).argsort()[:1])
    # ###############################  Clean  Up  ############################
    del exec_net
    del plugin
    # ########################################################################

import sys
if __name__ == '__main__':
    sys.exit(main() or 0)
