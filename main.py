import time
import sanelib
from lib.mdh import example as mdh_example

# Starting time
start = time.time()

# Run library
mdh = sanelib.mdh
mdh_example.run(mdh)

# End time
end = time.time()

# Total time taken
print(f"Runtime: {end - start} [s]")

