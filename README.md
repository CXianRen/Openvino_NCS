#Openvino & NCS 
we posit that you have installed tensorflow & Openvino and ran the demo of them correctly!
if not  there are some tips may help:
tensotflow:
	http://www.tensorflow.org/install 

Openvino for linux:
	https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html

OPenvino for raspberry:
	https://software.intel.com/en-us/articles/OpenVINO-Install-RaspberryPI
	https://software.intel.com/en-us/neural-compute-stick/get-started
	
more useful imformation you can get here:
	https://docs.openvinotoolkit.org/latest/_inference_engine_ie_bridges_python_docs_api_overview.html

#just 4 steps 
# 1、run: python3 mnist_train.py	
#it will create a .meta file 

# 2、run: python3 create.py		
#reload & resave the net to ./ncs 

# 3、run: mo_tf --input_meta_graph  ./ncs/cnn-mnist_inference.meta(the .meta file in ./ncs)  --batch 1 	
# if success, you will get a xxx.xml file & a xxx.bin file in ./ncs

# 4、 cd ./ncs then run: python3 run.py 
# if not thing wrong, you will get the FPS and the result of the 1.jpg/0.jpg
 
# how to create a useful net by yourself,you can get something here:
#is a example of the NCSDK1/2, and it work in Openvino! if you get more about this step, XD!
	https://movidius.github.io/ncsdk/tf_compile_guidance.html

# a sample of yolov3		
	https://github.com/PINTO0309/OpenVINO-YoloV3/blob/master/openvino_tiny-yolov3_test.py

some solutions of errors:
	https://software.intel.com/en-us/forums/computer-vision/topic/801113
	https://software.intel.com/en-us/articles/transitioning-from-intel-movidius-neural-compute-sdk-to-openvino-toolkit#inpage-nav-13

