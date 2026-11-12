import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.utils.paths import DATA_DIR

MCS_DIR = DATA_DIR / "mcd"
MCD_METADATA = MCS_DIR / "metadata.csv"

df = pd.read_csv(MCD_METADATA)

random_row = df.sample(n=1).iloc[0]

faces = np.load(MCS_DIR / random_row.frames_file)
ppg = np.load(MCS_DIR / random_row.ppg_file)

face = faces[0, ...]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.imshow(face)
ax2.plot(ppg[:300])
plt.show()


