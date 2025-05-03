import evaluator
import json

if __name__ == "__main__":
    # eval_model = evaluator.evaluation_model()
    # with open("./train.json") as f:
    #     labels = json.load(f)
    
    # print(type(labels))

    from PIL import Image
    import os

    for i in range(5):
        img = Image.open(f"images/test/{i}.png")
        img.show()
