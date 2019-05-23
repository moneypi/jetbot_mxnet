import mxnet
from collections import namedtuple
import cv2

Batch = namedtuple('Batch', ['data'])

sym, arg_params, aux_params = mxnet.model.load_checkpoint("jetbot_pre", 0)
model = mxnet.mod.Module(symbol=sym, label_names=['data'])
model.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
model.set_params(arg_params, aux_params, allow_missing=True)

img = cv2.imread("/home/luke/Download/face/data/hotdog/train/not-hotdog/9.png")
img = img[:, :, ::-1]  # bgr_to_rgb

img = mxnet.nd.array(img, dtype="uint8")
img = mxnet.gluon.data.vision.transforms.Resize(256).forward(img)
img = mxnet.gluon.data.vision.transforms.CenterCrop(224).forward(img)
img = mxnet.gluon.data.vision.transforms.ToTensor().forward(img)
img = mxnet.gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).forward(img)

model.forward(Batch([mxnet.nd.expand_dims(img, axis=0)]))
prob = model.get_outputs()[0].asnumpy()
print(prob)
