# ---- Step 13: CNN-LSTM (authors' second model), same folds as E-LSTM ----
import time
CNN_EPOCHS = 5            # same number of epochs as the E-LSTM LSTM (authors' repo: 1)
assert IN_STEP == 10, 'The authors\' CNN-LSTM reshape (8,32) is written for the 10-step window'

class CnnSeq(tf.keras.utils.Sequence):
    """Raw windows (10,6,500) built on the fly for the positions `pos` of all_idx."""
    def __init__(self, pos, bs=32, shuffle=False):
        super().__init__()
        self.pos, self.bs, self.shuffle = np.array(pos), bs, shuffle
        self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.pos) / self.bs))
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.pos)
    def __getitem__(self, b):
        p = self.pos[b * self.bs:(b + 1) * self.bs]
        x = np.stack([cells[all_idx[k][0]]['X'][all_idx[k][1]:all_idx[k][1] + IN_STEP] for k in p])
        return x, Y[p]

def make_cnn_lstm():
    m = models.Sequential([
        layers.Input((IN_STEP, 6, LEN)),
        layers.Conv2D(32, (3, 3), activation='relu'),     # as in the authors' notebook
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Reshape((8, 32)),                          # 8,32 for the 10-step window
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(OUT_STEP, activation='sigmoid')])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mae')   # lr 1e-3 for CNN-LSTM in the repo
    return m

# same folds as Step 11 (so the comparison with E-LSTM is fair)
if SPLIT_MODE == 'paper': splits = list(KFold(5, shuffle=True, random_state=42).split(E))
else:                     splits = list(GroupKFold(5).split(E, groups=grp))

res_cnn, cnn_models = [], []
for f, (tr, va) in enumerate(splits):
    t0 = time.time()
    m = make_cnn_lstm()
    m.fit(CnnSeq(tr, 32, True), epochs=CNN_EPOCHS, verbose=0)
    p = m.predict(CnnSeq(va, 256), verbose=0)
    mae = np.mean(np.abs(Y[va] - p)); mse = np.mean((Y[va] - p) ** 2)
    res_cnn.append([mae * 100, mse * 1e3, np.sqrt(mse) * 100, mape(Y[va], p)])
    cnn_models.append(m)
    print(f'CNN-LSTM fold {f+1}/5: MAE={res_cnn[-1][0]:.2f}e-2  MSE={res_cnn[-1][1]:.3f}e-3  '
          f'RMSE={res_cnn[-1][2]:.2f}e-2  MAPE={res_cnn[-1][3]:.2f}%   ({(time.time()-t0)/60:.1f} min)')
res_cnn = np.array(res_cnn)

assert len(res) == 5, 'Run Step 11 first (with the same SPLIT_MODE)'
paper_cnn = {(10, 10): [0.90, 0.22, 1.40, 0.96]}[(IN_STEP, OUT_STEP)]
paper_el  = PAPER[(IN_STEP, OUT_STEP)]
print(f'\n=== Window {(IN_STEP, OUT_STEP)}, split={SPLIT_MODE}: mean over 5 folds ===')
print(f'{"metric":12s}{"E-LSTM":>9s}{"(paper)":>9s}{"CNN-LSTM":>11s}{"(paper)":>9s}')
for n_, a, pa, b, pb in zip(['MAE x1e-2', 'MSE x1e-3', 'RMSE x1e-2', 'MAPE %'],
                             res.mean(0), paper_el, res_cnn.mean(0), paper_cnn):
    print(f'{n_:12s}{a:9.2f}{pa:9.2f}{b:11.2f}{pb:9.2f}')
print('std CNN-LSTM:', np.round(res_cnn.std(0), 2))
