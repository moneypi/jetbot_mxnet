import mxnet
from collections import namedtuple
import numpy as np
import cv2

Batch = namedtuple('Batch', ['data'])

sym, arg_params, aux_params = mxnet.model.load_checkpoint("jetbot_pre", 0)
model = mxnet.mod.Module(symbol=sym, label_names=['data'])
model.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
model.set_params(arg_params, aux_params, allow_missing=True)

img = cv2.imread("/home/luke/Download/face/d2l-zh/test_data/not-hotdog/0.png")
pic = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
start_pos = int((256 - 224)/2)
end_pos = int(start_pos + 224)
cropImg = pic[start_pos:end_pos, start_pos:end_pos]
# cv2.imshow("cropImg", cropImg)
# cv2.waitKey(10000)
normalize_pos = cropImg/255
normalize_pos = normalize_pos[:,:,::-1]
print(normalize_pos.shape)
# print(normalize_pos[0])
bgr_mean = np.array([0.485, 0.456, 0.406])
mean_pic = normalize_pos - bgr_mean
print(mean_pic.shape)
# print(mean_pic)
bgr_std = np.array([0.229, 0.224, 0.225])
std_pic = mean_pic/bgr_std
print(std_pic.shape)
final_pic = np.zeros((3, 224, 224))
final_pic[0] = std_pic[:,:,0]
final_pic[1] = std_pic[:,:,1]
final_pic[2] = std_pic[:,:,2]
# print(final_pic)

#
expand_pic = np.expand_dims(final_pic, axis=0)
# print(expand_pic.shape)
model.forward(Batch([mxnet.nd.array(expand_pic)]))
prob = model.get_outputs()[0].asnumpy()
print(prob)
