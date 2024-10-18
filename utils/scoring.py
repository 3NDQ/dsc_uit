import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score

reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'

print('Reading prediction')
submission = json.load(open(os.path.join(prediction_dir, "results.json")))
phase = submission["phase"]
if phase == "dev":
    ref_file = "dev.json"
elif phase == "test":
    ref_file = "test.json"
else:
    print(f"Phase {phase} is invalid.")
refs = json.load(open(os.path.join(reference_dir, ref_file)))

assert submission["phase"] == refs["phase"]
assert submission["results"].keys() == refs["results"].keys()

submission = list(submission["results"].values())
refs = list(refs["results"].values())

print('Checking Precision')
precision = precision_score(refs, submission, average="macro")
print('Checking Recall')
recall = recall_score(refs, submission, average="macro")
print('Checking F1 score')
f1 = f1_score(refs, submission, average="macro")

print('Scores:')
scores = {
    'precision': precision,
    "recall": recall,
    "f1": f1
}
print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
