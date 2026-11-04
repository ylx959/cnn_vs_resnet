import torch
from PIL import Image

from cnn import BasicCNN, PlainCNN
from data import get_transforms, split_indices
from resnet import ResNet20


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_plain_and_resnet_have_same_params_and_init():
    torch.manual_seed(0)
    plain = PlainCNN()
    torch.manual_seed(0)
    resnet = ResNet20()
    assert count_params(plain) == count_params(resnet) == 269_722
    for a, b in zip(plain.state_dict().values(), resnet.state_dict().values()):
        assert torch.equal(a, b)


def test_skip_connection_changes_output():
    torch.manual_seed(0)
    plain = PlainCNN().eval()
    torch.manual_seed(0)
    resnet = ResNet20().eval()
    x = torch.randn(2, 3, 32, 32)
    assert plain(x).shape == resnet(x).shape == (2, 10)
    assert not torch.allclose(plain(x), resnet(x))


def test_basic_cnn():
    model = BasicCNN()
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert count_params(model) == 4_293_194
    assert sum(isinstance(m, torch.nn.MaxPool2d) for m in model.modules()) == 2


def test_split_is_disjoint_and_stable():
    train, val = split_indices()
    assert len(train) == 45000 and len(val) == 5000
    assert not set(train) & set(val)
    assert split_indices() == (train, val)


def test_only_train_transform_is_random():
    image = Image.fromarray(torch.randint(0, 255, (32, 32, 3), dtype=torch.uint8).numpy())
    transform_train, transform_test = get_transforms()

    assert torch.equal(transform_test(image), transform_test(image))
    outputs = [transform_train(image) for _ in range(5)]
    assert outputs[0].shape == (3, 32, 32)
    assert any(not torch.equal(outputs[0], o) for o in outputs[1:])
