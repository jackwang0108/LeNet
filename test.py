from tqdm import tqdm, trange
from time import sleep

# bar = trange(10)
# for i in bar:
#     # Print using tqdm class method .write()
#     sleep(0.1)
#     if not (i % 3):
#         tqdm.write(f"Done task {i}")
    # Can also use bar.write()


for i in tqdm(range(3)):
    for j in tqdm(range(5), leave=False):
        sleep(1)



# outer = trange(3)
# inner = tqdm(range(5), leave=False)
#
# for i in outer:
#     # outer.write(f"outer: {i}")
#     for j in inner:
#         # inner.write(f"\tinner: {j}")
#         sleep(0.1)