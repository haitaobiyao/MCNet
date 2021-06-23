from network import MCNet
from utils.pyt_utils import load_model
from tools.benchmark.compute_speed import compute_speed
from tools.benchmark.compute_flops import compute_flops
from tools.benchmark.flops_params_count import get_model_complexity_info

if __name__ == "__main__":
    network = MCNet(10, criterion=None, edge_criterion=None)
    model_file = "/home/xionghaitao/workplace/segmantic_segmentation/TorchSeg/log/scut.ernet.R101/snapshot/epoch-49.pth"
    model = load_model(network, model_file)
    model = model.cuda()
    model.eval()
    device = 0
    flops_count, params_count = get_model_complexity_info(model, (3, 576, 720))
