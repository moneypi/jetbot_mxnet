import mxnet
from collections import namedtuple

sym, arg_params, aux_params = mxnet.model.load_checkpoint("jetbot_pre", 0)
model = mxnet.mod.Module(symbol=sym, label_names=['data'])
model.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
model.set_params(arg_params, aux_params, allow_missing=True)

normalize = mxnet.gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

test_augs = mxnet.gluon.data.vision.transforms.Compose([
    mxnet.gluon.data.vision.transforms.Resize(256),
    mxnet.gluon.data.vision.transforms.CenterCrop(224),
    mxnet.gluon.data.vision.transforms.ToTensor(),
    normalize])

pre_imgs = mxnet.gluon.data.vision.ImageFolderDataset('/home/luke/Download/face/d2l-zh/test_data/')

predict_iter = mxnet.gluon.data.DataLoader(
    pre_imgs.transform_first(test_augs), 1)

Batch = namedtuple('Batch', ['data'])

k = 0
for X, _ in predict_iter:
    print(type(X))
    print(X.shape)
    model.forward(Batch([X]))
    prob = model.get_outputs()[0].asnumpy()
    print(prob)
    k = k + 1
    if k > 10:
        break
