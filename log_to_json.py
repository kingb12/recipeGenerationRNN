import sys
import json

def read_log(log):
    """
    reads a log file and returns lists of epoch, training loss, learning rate
    """
    epochs = []
    losses = []
    lrs = []
    t_perps = []
    vlosses = []
    v_perps= []
    if sys.argv[-1] == "True":
        euclid_means = []
        euclid_stds = []
        cosine_means = []
        cosine_stds = []
        euclid_kmeans = []
        euclid_kstds = []
        cosine_kmeans = []
        cosine_kstds = []
    with open(log, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
      if sys.argv[-1] == "True":
            epoch, loss, lr, t_perp, vloss, v_perp, e_m, e_std, c_m, c_std, e_km, e_kstd, c_km, c_kstd, _ = line.split('\t')
            euclid_means.append(e_m)
            euclid_stds.append(e_std)
            cosine_means.append(e_km)
            cosine_stds.append(e_kstd)
            euclid_kmeans.append(c_m)
            euclid_kstds.append(c_std)
            cosine_kmeans.append(c_km)
            cosine_kstds.append(c_kstd)
      else:
          epoch, loss, lr, t_perp, vloss, v_perp, _ = line.split('\t')
      epochs.append(float(epoch))
      losses.append(float(loss))
      lrs.append(float(lr))
      vlosses.append(float(vloss))
      v_perps.append(float(v_perp))
      t_perps.append(float(t_perp))
    if sys.argv[-1] == "True":
        return epochs, losses, lrs, t_perps, vlosses, v_perps, euclid_means, euclid_stds, cosine_means, cosine_stds, euclid_kmeans, euclid_kstds, cosine_kmeans, cosine_kstds

    else:
        return epochs, losses, lrs, t_perps, vlosses, v_perps


epochs = []
losses = []
lrs = []
t_perps = []
vlosses = []
v_perps= []
if sys.argv[-1] == "True":
     euclid_means = []
     euclid_stds = []
     cosine_means = []
     cosine_stds = []
     euclid_kmeans = []
     euclid_kstds = []
     cosine_kmeans = []
     cosine_kstds = []
e_c = 0
for i, log_file in enumerate(sys.argv[2:-1]):
    if sys.argv[-1] == "True":
        e, lo, lr, t_p, vlo, v_p, e_m, e_std, c_m, c_std, e_km, e_kstd, c_km, c_kstd  = read_log(log_file)
        euclid_means.extend(e_m)
        euclid_stds.extend(e_std)
        cosine_means.extend(e_km)
        cosine_stds.extend(e_kstd)
        euclid_kmeans.extend(c_m)
        euclid_kstds.extend(c_std)
        cosine_kmeans.extend(c_km)
        cosine_kstds.extend(c_kstd)
    else:
        e, lo, lr, t_p, vlo, v_p = read_log(log_file)
    t_px = []
    for t in t_p:
        t_px.append(t)
    epochs.extend([n + (e_c) for n in e])
    e_c += len(e)
    losses.extend(lo)
    vlosses.extend(vlo)
    lrs.extend(lr)
    t_perps.extend(t_px)
    v_perps.extend(v_p)


i = 0
lr = lrs[0]
if sys.argv[-1] == "True":
    result = {lr: [list(),list(),list(),list(), list(),list(),list(),list(), list(),list(),list(),list(), list()]}
else:
    result = {lr: [list(),list(),list(),list(), list()]}
while i < len(epochs):
    if lrs[i] == lr:
        result[lr][0].append(epochs[i])
        result[lr][1].append(losses[i])
        result[lr][2].append(vlosses[i])
        result[lr][3].append(t_perps[i])
        result[lr][4].append(v_perps[i])
        if sys.argv[-1] == "True":
            result[lr][5].append(euclid_means[i])
            result[lr][6].append(euclid_stds[i])
            result[lr][7].append(cosine_means[i])
            result[lr][8].append(cosine_stds[i])
            result[lr][9].append(euclid_kmeans[i])
            result[lr][10].append(euclid_kstds[i])
            result[lr][11].append(cosine_kmeans[i])
            result[lr][12].append(cosine_kstds[i])
            
        i = i + 1
    else:
        lr = lrs[i]
        result[lr] = [list(), list(),list(),list(),list()]

with open(sys.argv[1], 'w') as f:
    f.write(json.dumps(result))
    
