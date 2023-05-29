import json
import os


def cat_effect(effect_root: str, keys: list, save_path: str = None):
    effect_path = []
    effect_names = os.listdir(effect_root)
    for name in effect_names:
        effect_path.append(os.path.join(effect_root, name))

    effects = dict()
    for key in keys:
        if key == 'epoch':
            effects[key] = 0
            continue
        effects[key] = [[], []]

    for effect in effect_path:
        with open(effect, 'r', encoding='utf-8') as f:
            e = json.load(f)

        for key in e.keys():
            if key == 'epoch':
                effects[key] += e[key]
                continue
            effects[key][0].extend(e[key][0])
            effects[key][1].extend(e[key][1])

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(effects))
        print('Successfully save effect.json in {}'.format(save_path))

    return effects


if __name__ == '__main__':
    effect_root = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\runs'
    save_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\runs\effect.json'
    keys = ['epoch', 'loss', 'acc', 'precision', 'recall', 'f1']

    effect = cat_effect(effect_root, keys, save_path)
    print(len(effect['precision']))
    print(len(effect['recall']))
    print(len(effect['f1']))
