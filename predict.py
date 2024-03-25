import torch
from core import Log1DNetv2, Log1DNetv3
import numpy as np

def load_model(modelpath:str='/content/best_model.pt'):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Log1DNetv3().to(device)
    model.load_state_dict(torch.load(modelpath))

    return model


def predict_logs(x, y, original_df):
    model = load_model()

    inputs = torch.from_numpy(x.values).cuda().float()
    # outputs = torch.from_numpy(y.values).cuda().float()

    dtcs = []; dtss = []
    model.eval()
    for i in range(inputs.shape[0]):
        with torch.inference_mode():
            pred = model(inputs[i].unsqueeze(0).repeat(64, 1, 1))
            dtc = pred[0][0]; dts = pred[0][1]
            dtcs.append(dtc.cpu().numpy()); dtss.append(dts.cpu().numpy())

    df = original_df.copy()
    df['Predicted DTC'] = np.array(dtcs).astype(float)
    df['Predicted DTS'] = np.array(dtss).astype(float)

    return df