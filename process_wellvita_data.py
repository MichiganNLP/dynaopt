import csv, json
csv_file = "data/wellvita_dataset.csv"
csv_reader = csv.reader(open(csv_file, encoding="utf-8"))
data = {}
header = next(csv_reader)
for i, row in enumerate(csv_reader):
    dict = {}
    for j in range(len(header)):
        dict[header[j]] = row[j]
    dialog_id = dict["dialog_id"]
    if dialog_id not in data:
        data[dialog_id] = []
    data[dialog_id].append(dict)
pairs = []
cr, sr = 0,0
num_words = []
for id, utts in data.items():
    for i, utt in enumerate(utts):
        if utt["author"] == "speaker":
            prompt_dict = utt
        else:
            continue
        for j in range(i+1, len(utts)):
            if utts[j]["author"] == "listener":
                label = utts[j]["final agreed label"] 
                if label == "Complex Reflection" or label == "Simple Reflection":
                    if label == "Complex Reflection":
                        cr += 1
                    else:
                        sr += 1
                    response_dict = utts[j]
                    pair = {
                                    "prompt": prompt_dict["text"],
                                    "response": response_dict["text"],
                                    "label": label,
                                }
                    pair.update(response_dict)
                    pairs.append(pair)
                    num_words.append(len(pair["prompt"].split()))
                    num_words.append(len(pair["response"].split()))
                    break
with open("data/wellvita_dataset.json", "w") as f:
    json.dump(pairs, f, indent=4)
print(len(pairs))
print("CR: ", cr)
print("SR: ", sr)
print("Avg. num words: ", sum(num_words)/len(num_words))