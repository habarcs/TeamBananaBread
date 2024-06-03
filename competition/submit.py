import requests
import json

import torch

from trainer import DEVICE


def submit(results, url="https://competition-production.up.railway.app/results/"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


def competition_test_loop(model, dataloader, classes):
    model.eval()
    preds = {}
    with torch.no_grad():
        for img, img_id in dataloader:
            img = img.to(DEVICE)
            pred = model(img)
            if isinstance(pred, tuple):
                pred = pred[-1]
            pred_id = pred.argmax(1).item()
            preds[img_id] = classes[pred_id].split('_')[0]

    res = {
        "images": preds,
        "groupname": "TeamBananaBread"
    }
    submit(res)
