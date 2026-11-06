"""Train a multinomial Naive Bayes scam/spam classifier on the UCI SMS Spam Collection
and export it as compact JSON for the G-SquadCare demo (JavaScript).
Tokeniser is deliberately simple so it can be replicated exactly in JS."""
import re, json, math, random, sys
from collections import Counter

SRC = sys.argv[1]; OUT = sys.argv[2]
rows = []
for line in open(SRC, encoding='utf8'):
    lab, _, txt = line.rstrip('\n').partition('\t')
    if lab in ('ham', 'spam'): rows.append((1 if lab == 'spam' else 0, txt))

def tokens(text):
    t = text.lower()
    feats = []
    if re.search(r'https?://|www\.|bit\.ly|\.com\b|\.my\b|\.ly\b', t): feats.append('<url>')
    if re.search(r'\b\d{5,}\b', t): feats.append('<longnum>')
    if re.search(r'(rm|\$|£|usd|myr)\s?\d|\b\d+\s?(rm|ringgit|dollars?|pounds?)\b|£|\$', t): feats.append('<money>')
    if re.search(r'\d{1,3}\s?%', t): feats.append('<percent>')
    if re.search(r'!{1,}', t): feats.append('<exclaim>')
    if any(ch.isupper() for ch in text) and sum(ch.isupper() for ch in text) >= 8: feats.append('<caps>')
    words = re.findall(r"[a-z][a-z']+", t)
    feats += [w for w in words if len(w) >= 2]
    return set(feats)   # binary bag of words (Bernoulli-style presence features)

random.seed(42)
idx = list(range(len(rows))); random.shuffle(idx)
spam_idx = [i for i in idx if rows[i][0] == 1]; ham_idx = [i for i in idx if rows[i][0] == 0]
def split(ids, k): n = int(len(ids) * 0.8); return ids[:n], ids[n:]
tr_s, te_s = split(spam_idx, 0); tr_h, te_h = split(ham_idx, 0)
train = [rows[i] for i in tr_s + tr_h]; test = [rows[i] for i in te_s + te_h]

def fit(data, min_count=2):
    df = [Counter(), Counter()]; n = [0, 0]
    for y, txt in data:
        n[y] += 1
        for f in tokens(txt): df[y][f] += 1
    vocab = {f for c in df for f in c if df[0][f] + df[1][f] >= min_count}
    alpha = 1.0
    # multinomial-style over presence features with Laplace smoothing
    tot = [sum(df[y][f] for f in vocab) + alpha * len(vocab) for y in (0, 1)]
    llr = {}
    for f in vocab:
        p1 = (df[1][f] + alpha) / tot[1]; p0 = (df[0][f] + alpha) / tot[0]
        llr[f] = math.log(p1) - math.log(p0)
    prior = math.log(n[1] / n[0])
    return llr, prior

def predict(llr, prior, txt):
    s = prior + sum(llr.get(f, 0.0) for f in tokens(txt))
    return 1 / (1 + math.exp(-s))

def metrics(model, data, thr=0.5):
    llr, prior = model
    tp = fp = fn = tn = 0
    for y, txt in data:
        p = predict(llr, prior, txt) >= thr
        if p and y: tp += 1
        elif p and not y: fp += 1
        elif (not p) and y: fn += 1
        else: tn += 1
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0,
                accuracy=(tp + tn) / len(data), tp=tp, fp=fp, fn=fn, tn=tn)

model = fit(train)
m = metrics(model, test)
print('held-out test (n=%d, spam=%d):' % (len(test), len(te_s)), {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

# 5-fold cross-validation on all data
folds = [[] for _ in range(5)]
for j, i in enumerate(spam_idx): folds[j % 5].append(rows[i])
for j, i in enumerate(ham_idx): folds[j % 5].append(rows[i])
cv = []
for k in range(5):
    tr = [r for j, f in enumerate(folds) if j != k for r in f]
    cv.append(metrics(fit(tr), folds[k]))
print('5-fold CV mean precision %.3f recall %.3f f1 %.3f' % tuple(sum(c[x] for c in cv) / 5 for x in ('precision', 'recall', 'f1')))

# demo messages
demo = ["Ini Inspektor Rahman dari PDRM. Akaun anda terlibat dalam siasatan pengubahan wang haram. Jangan beritahu sesiapa, termasuk anak anda. Pindahkan simpanan anda ke akaun selamat hari ini.",
        "Hi Auntie, join our VIP investment group. Guaranteed 30% returns in 7 days! Transfer RM500 to secure your slot.",
        "Peringatan: temujanji anda esok, 9:30 pagi. Sila bawa MyKad. Reminder: your appointment is tomorrow at 9:30 am.",
        "Your parcel is held by customs. Pay RM8.50 today at bit.ly/kastam-my or legal action will be taken.",
        "This is Inspector Rahman from PDRM. Your account is under investigation for money laundering. Do not tell anyone. Transfer your savings to a safe account today.",
        "Mum, I'll be home for dinner at 7. Can you keep some rice for me?",
        "Your Maybank account will be suspended. Verify now at www.maybank-secure.com to avoid suspension."]
llr, prior = model
for d in demo:
    toks = tokens(d); cov = sum(1 for f in toks if f in llr) / max(1, len(toks))
    print('%.3f  cov %.2f  %s' % (predict(llr, prior, d), cov, d[:60]))

# export: keep features with |llr| >= 0.3 to shrink the file
export = {f: round(v, 2) for f, v in llr.items()}
json.dump({'prior': round(prior, 4), 'llr': export, 'n_train': len(train), 'vocab': len(llr), 'kept': len(export),
           'test': {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}, open(OUT, 'w'))
print('exported', len(export), 'of', len(llr), 'features ->', OUT)
