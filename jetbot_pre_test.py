import mxnet
from collections import namedtuple
import cv2

img = cv2.imread("/home/luke/Download/face/data/hotdog/train/not-hotdog/9.png")
img = img[:, :, ::-1]  # bgr_to_rgb

img = mxnet.nd.array(img, dtype="uint8")
img = mxnet.gluon.data.vision.transforms.Resize(256).forward(img)
img = mxnet.gluon.data.vision.transforms.CenterCrop(224).forward(img)
img = mxnet.gluon.data.vision.transforms.ToTensor().forward(img)
img = mxnet.gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).forward(img)
# print(img[0,0,:])
img = mxnet.nd.expand_dims(img, axis=0)
sym = mxnet.sym.load('jetbot_pre-symbol.json')
data = mxnet.sym.var('data')
net = mxnet.gluon.SymbolBlock(sym, data)
net.load_parameters('jetbot_pre-0000.params')
res = net(img)
print(res)

all_layer = sym.list_arguments()
out_layer = sym.list_outputs()
# print(all_layer)
all_internal = sym.get_internals()
# list_all = [all_internal[x] for x in all_layer]
middle_output = [all_internal['resnetv21_dense0_bias']]
new_net = mxnet.gluon.SymbolBlock(middle_output, data, params=net.collect_params())
res = new_net(img)
print(res)
